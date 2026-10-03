#include <spdlog/sinks/ostream_sink.h>

#include <cmath>
#include <filesystem>
#include <macis/doping/charge_sectors.hpp>
#include <macis/doping/fix_mu.hpp>
#include <macis/impurity_solver.hpp>
#include <macis/parity_sectors.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/wavefunction_io.hpp>
#include <sstream>

#include "ut_common.hpp"

// Band-parity sector solve (parity-sector-solve-simple.md, §7).

namespace {

constexpr size_t NB = 64;
using params_t = macis::impurity_params<NB>;

// The two-band Kanamori model of tests/charge_sectors.cxx -- impurity orbitals
// 0 (band 0) and 1 (band 1), bath levels 2..5 coupled to band k % 2 -- with
// the cross-band hybridization `cross` (0.15 there) as a parameter. At
// cross = 0 the bath is band-diagonal and both band parities are conserved.
// `degenerate` makes the two bands identical (bath levels -2.0, -2.0, 0.6,
// 0.6), so that band swap is an exact symmetry.
struct ModelOpts {
  size_t na = 3, nb = 3;
  double cross = 0.0;
  double eps_d = -3.0;
  double pair_hopping = 0.8;  // J_P; 0 makes the band counts conserved
  bool degenerate = false;
  CIExpansion ci = CIExpansion::ASCI;
  size_t ntdets_max = 1000;  // every parity sector has <= 200 determinants
};

params_t make_model(const ModelOpts& o) {
  const size_t n_imp = 2, n = 6;
  const double U = 6.0, J = 0.8;
  params_t p{};
  p.norb = p.n_active = n;
  p.n_inactive = 0;
  p.n_imp = n_imp;
  p.nbands = 2;
  p.nalpha = o.na;
  p.nbeta = o.nb;
  p.E_core = 0.0;
  p.E_inactive = 0.0;
  p.just_singles = false;
  p.spin_dep = false;
  p.ci_exp = o.ci;
  p.T.assign(n * n, 0.0);
  p.Td.assign(n * n, 0.0);
  p.V.assign(n * n * n * n, 0.0);

  const double eps_nd[4] = {-2.0, -0.7, 0.6, 1.9};
  const double eps_dg[4] = {-2.0, -2.0, 0.6, 0.6};
  for(size_t k = 0; k < 4; ++k) {
    const size_t b = n_imp + k, band = k % 2, other = 1 - band;
    p.T[b + b * n] = o.degenerate ? eps_dg[k] : eps_nd[k];
    p.T[band + b * n] = p.T[b + band * n] = 0.5;
    p.T[other + b * n] = p.T[b + other * n] = o.cross;
  }
  for(size_t m = 0; m < n_imp; ++m) p.T[m + m * n] = o.eps_d;
  auto V = [&](size_t a, size_t b, size_t c, size_t d) -> double& {
    return p.V[a + b * n + c * n * n + d * n * n * n];
  };
  for(size_t m = 0; m < n_imp; ++m) {
    V(m, m, m, m) = U;
    for(size_t mp = 0; mp < n_imp; ++mp) {
      if(mp == m) continue;
      V(m, m, mp, mp) = U - 2 * J;
      V(m, mp, mp, m) = J;
      V(m, mp, m, mp) = o.pair_hopping;
    }
  }

  p.mcscf_settings.ci_max_subspace = 200;
  p.mcscf_settings.ci_res_tol = 1e-10;
  p.asci_settings.ntdets_max = o.ntdets_max;
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

  p.dstep = 2e-2;
  p.abs_tol = 1e-7;
  p.maxiter = 100;
  p.print_doping = false;
  p.init_shift = 2.0;
  p.delta_CFS = 0.0;
  p.cheap_mode = false;
  return p;
}

void build_active(params_t& p) {
  const size_t n = p.norb;
  macis::active_hamiltonian(NumOrbital(n), NumActive(n), NumInactive(0),
                            p.T.data(), n, p.V.data(), n, p.F_inactive.data(),
                            n, p.T_active.data(), n, p.V_active.data(), n);
}

// Model with the parity solve switched on (labels built before the active
// integrals, as in the driver)
params_t make_parity_model(const ModelOpts& o) {
  auto p = make_model(o);
  macis::setup_parity_sectors<NB>(p, 1e-10);
  build_active(p);
  return p;
}

uint32_t key_of(const params_t& p, const macis::wfn_t<NB>& d) {
  return macis::ParityMasks<NB>(*p.parity_labels).key(d);
}

// Exact sector minima of the cross = 0 model (parity-sector-toy-ed.py)
constexpr double E33_ee = -8.512442367, E33_oo = -8.551448456;
constexpr double E32_eo = -8.700141181, E32_oe = -8.865079189;
constexpr uint32_t KEY_EE = 0b00, KEY_OE = 0b01, KEY_EO = 0b10, KEY_OO = 0b11;

// Capture std::cout for the duration of a scope
struct CoutCapture {
  std::ostringstream buf;
  std::streambuf* old;
  CoutCapture() : old(std::cout.rdbuf(buf.rdbuf())) {}
  ~CoutCapture() { std::cout.rdbuf(old); }
  std::string str() const { return buf.str(); }
};

// Capture the asci_search logger (spdlog writes to stdout directly, so
// CoutCapture does not see it): the PARITY FILTER lines are logged there
struct SearchLogCapture {
  std::ostringstream buf;
  std::shared_ptr<spdlog::logger> old = spdlog::get("asci_search");
  SearchLogCapture() {
    spdlog::drop("asci_search");
    spdlog::register_logger(std::make_shared<spdlog::logger>(
        "asci_search", std::make_shared<spdlog::sinks::ostream_sink_mt>(buf)));
  }
  ~SearchLogCapture() {
    spdlog::drop("asci_search");
    if(old) spdlog::register_logger(old);
  }
  std::string str() const { return buf.str(); }
};

bool root_rank() {
  int rank = 0;
  MACIS_MPI_CODE(MPI_Comm_rank(MPI_COMM_WORLD, &rank);)
  return rank == 0;
}
void barrier() { MACIS_MPI_CODE(MPI_Barrier(MPI_COMM_WORLD);) }

// One scratch directory shared by all ranks: the solver writes seed files
// on rank 0 and every rank reads them back
struct ScratchDir {
  std::filesystem::path old = std::filesystem::current_path();
  std::filesystem::path dir;
  ScratchDir(const std::string& tag) {
    dir = std::filesystem::temp_directory_path() / ("macis_parity_" + tag);
    if(root_rank()) {
      std::filesystem::remove_all(dir);
      std::filesystem::create_directories(dir);
    }
    barrier();
    std::filesystem::current_path(dir);
  }
  ~ScratchDir() {
    barrier();
    std::filesystem::current_path(old);
  }
};

template <typename... Args>
void write_wfn_root(const std::string& fname, Args&&... args) {
  if(root_rank()) macis::write_wavefunction(fname, args...);
  barrier();
}

}  // namespace

TEST_CASE("Parity labels") {
  SECTION("band-diagonal bath: two groups, nothing discarded") {
    auto p = make_parity_model({});
    const auto& L = *p.parity_labels;
    REQUIRE(L.ngroups == 2);
    CHECK(L.group_orbs[0] == std::vector<uint32_t>{0, 2, 4});
    CHECK(L.group_orbs[1] == std::vector<uint32_t>{1, 3, 5});
    CHECK(L.max_discarded == 0.0);
    CHECK_FALSE(L.counts_conserved);
  }

  SECTION("cross-band hybridization is refused") {
    ModelOpts o;
    o.cross = 0.15;
    auto p = make_model(o);
    CHECK_THROWS_WITH(macis::setup_parity_sectors<NB>(p, 1e-10),
                      Catch::Contains("not band-diagonal"));
  }

  SECTION("cross-band noise below PARITY_TOL is zeroed and reported") {
    ModelOpts o;
    o.cross = 1e-12;
    auto p = make_model(o);
    macis::setup_parity_sectors<NB>(p, 1e-10);
    CHECK(p.parity_labels->max_discarded == 1e-12);
    for(size_t q = 0; q < p.norb; ++q)
      for(size_t r = 0; r < p.norb; ++r)
        if(p.parity_labels->group_of[q] != p.parity_labels->group_of[r])
          CHECK(p.T[q + r * p.norb] == 0.0);
  }

  SECTION("a parity-breaking interaction is refused") {
    auto p = make_model({});
    p.V[0 + 0 * 6 + 0 * 36 + 1 * 216] = 0.1;  // (00|01)
    CHECK_THROWS_WITH(macis::setup_parity_sectors<NB>(p, 1e-10),
                      Catch::Contains("changes a band parity"));
  }

  SECTION("without pair hopping the counts are conserved (warned)") {
    ModelOpts o;
    o.pair_hopping = 0.0;
    auto p = make_parity_model(o);
    CHECK(p.parity_labels->counts_conserved);
  }

  SECTION("GROW_WITH_ROT is refused, NROTS > 0 is not") {
    auto p = make_model({});
    p.asci_settings.grow_with_rot = true;
    CHECK_THROWS_WITH(macis::setup_parity_sectors<NB>(p, 1e-10),
                      Catch::Contains("GROW_WITH_ROT = FALSE"));
    auto q = make_model({});
    q.asci_settings.nrots = 2;
    CHECK_NOTHROW(macis::setup_parity_sectors<NB>(q, 1e-10));
  }

  SECTION("sector enumeration") {
    auto p = make_parity_model({});
    CHECK(macis::enumerate_parity_keys(*p.parity_labels, 6) ==
          std::vector<uint32_t>{KEY_EE, KEY_OO});
    CHECK(macis::enumerate_parity_keys(*p.parity_labels, 5) ==
          std::vector<uint32_t>{KEY_OE, KEY_EO});
  }
}

TEST_CASE("Parity seed") {
  auto p = make_parity_model({});
  const auto& L = *p.parity_labels;
  macis::SDBuildHamiltonianGenerator<NB> H(
      macis::matrix_span<double>(p.T_active.data(), 6, 6),
      macis::rank4_span<double>(p.V_active.data(), 6, 6, 6, 6));
  // Energy-ordered (3,3) reference: orbitals 0, 1 (eps_d = -3) and 2 (-2.0)
  const auto base = macis::canonical_hf_determinant<NB>(3, 3);
  REQUIRE(key_of(p, base) == KEY_EE);

  SECTION("the home sector keeps the base determinant") {
    CHECK(macis::parity_seed(base, KEY_EE, L, H, p.n_imp) == base);
  }

  SECTION("another sector: repaired, then a local minimum of <D|H|D>") {
    const auto d = macis::parity_seed(base, KEY_OO, L, H, p.n_imp);
    REQUIRE(key_of(p, d) == KEY_OO);
    const double E = H.matrix_element(d, d);
    // No sector-preserving single move lowers the diagonal energy
    for(const auto& m : macis::detail::grouped_singles(d, L)) {
      const auto d1 = macis::detail::apply_move(d, m);
      if(key_of(p, d1) == KEY_OO) CHECK(H.matrix_element(d1, d1) >= E - 1e-12);
    }
  }
}

TEST_CASE("Parity sectors: each sector against exact diagonalization") {
  ScratchDir scratch("each");
  struct Case {
    size_t na, nb;
    std::vector<int> only;
    double E;
  };
  const std::vector<Case> cases = {{3, 3, {0, 0}, E33_ee},
                                   {3, 3, {1, 1}, E33_oo},
                                   {3, 2, {0, 1}, E32_eo},
                                   {3, 2, {1, 0}, E32_oe}};
  for(const auto& c : cases) {
    for(auto ci : {CIExpansion::ASCI, CIExpansion::CAS}) {
      ModelOpts o;
      o.na = c.na;
      o.nb = c.nb;
      o.ci = ci;
      auto p = make_parity_model(o);
      p.parity_only = c.only;
      const double E = ci == CIExpansion::CAS
                           ? macis::SolveImpurityED<NB>(p)
                           : macis::SolveImpurityASCI_rot<NB>(p);
      CHECK(E == Approx(c.E).margin(1e-8));
      const uint32_t want = uint32_t(c.only[0]) | uint32_t(c.only[1]) << 1;
      for(const auto& d : p.dets) CHECK(key_of(p, d) == want);
    }
  }
  // The NROTS = 0 solver without the macro loop (used by Mu_vs_n)
  auto p = make_parity_model({});
  CHECK(macis::SolveImpurityASCI<NB>(p) == Approx(E33_oo).margin(1e-8));
}

TEST_CASE("Parity sectors: full solve") {
  ScratchDir scratch("full");

  SECTION("(3,3): the legacy solve is the (e,e) sector, bit for bit") {
    auto legacy = make_model({});
    build_active(legacy);
    const double E_legacy = macis::SolveImpurityASCI_rot<NB>(legacy);
    CHECK(E_legacy == Approx(E33_ee).margin(1e-8));

    auto ee = make_parity_model({});
    ee.parity_only = {0, 0};
    const double E_ee = macis::SolveImpurityASCI_rot<NB>(ee);
    CHECK(E_ee == E_legacy);
    CHECK(ee.dets == legacy.dets);
    CHECK(ee.C == legacy.C);
  }

  SECTION("(3,3): the wrapper returns (o,o), 0.039 Ha below the legacy") {
    auto p = make_parity_model({});
    SearchLogCapture search_log;
    CoutCapture cap;
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    const auto out = cap.str();
    const auto slog = search_log.str();
    CHECK(E == Approx(E33_oo).margin(1e-8));
    for(const auto& d : p.dets) CHECK(key_of(p, d) == KEY_OO);
    CHECK(out.find("PARITY_COVERAGE COMPLETE") != std::string::npos);
    CHECK(slog.find("PARITY FILTER (e,e): dropped 0 candidates") !=
          std::string::npos);
    CHECK(slog.find("PARITY FILTER (o,o): dropped 0 candidates") !=
          std::string::npos);
    CHECK(slog.find("dropped 1") == std::string::npos);
    CHECK(std::filesystem::exists("parity_sectors.dat"));
    // The wrapper leaves no sector state behind in p
    const bool target_cleared = !p.asci_settings.parity_target;
    CHECK(target_cleared);
    // occs belong to the winner's state
    double nel = 0.;
    for(auto o : p.occs) nel += 2 * o;
    CHECK(nel == Approx(6.0).margin(1e-8));
    CHECK(std::filesystem::exists("active_ordm.dat"));
  }

  SECTION("(3,2): the wrapper returns (o,e)") {
    ModelOpts o;
    o.nb = 2;
    auto p = make_parity_model(o);
    CHECK(macis::SolveImpurityASCI_rot<NB>(p) == Approx(E32_oe).margin(1e-8));
  }

  SECTION("CAS: the wrapper returns the global ground state") {
    ModelOpts o;
    o.ci = CIExpansion::CAS;
    auto p = make_parity_model(o);
    CHECK(macis::SolveImpurityED<NB>(p) == Approx(E33_oo).margin(1e-8));
  }

  SECTION("truncated budget: the filter drops nothing, (o,o) still wins") {
    ModelOpts o;
    o.ntdets_max = 60;
    auto p = make_parity_model(o);
    p.asci_settings.ntdets_min = 5;
    p.asci_settings.ncdets_max = 30;
    SearchLogCapture search_log;
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    const auto slog = search_log.str();
    CHECK(slog.find("dropped 0 candidates") != std::string::npos);
    CHECK(slog.find("dropped 1") == std::string::npos);
    CHECK(E >= E33_oo - 1e-8);
    CHECK(key_of(p, p.dets[0]) == KEY_OO);
  }
}

TEST_CASE("Parity sectors: guess wavefunctions") {
  ScratchDir scratch("guess");

  SECTION("a guess in one sector seeds that sector, the other starts cold") {
    auto ref = make_parity_model({});
    ref.parity_only = {1, 1};
    const double E_ref = macis::SolveImpurityASCI_rot<NB>(ref);
    write_wfn_root("oo.wfn", ref.n_active, ref.dets, ref.C);

    auto p = make_parity_model({});
    p.asci_wfn_fname = "oo.wfn";
    p.compute_asci_E0 = false;
    p.asci_E0 = E_ref;
    CoutCapture cap;
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    const auto out = cap.str();
    CHECK(E == Approx(E33_oo).margin(1e-8));
    CHECK(out.find("guess oo.wfn lies in sector (o,o)") != std::string::npos);
    CHECK(out.find("start=guess") != std::string::npos);
    CHECK(out.find("start=cold") != std::string::npos);
    CHECK(p.asci_wfn_fname == "oo.wfn");  // restored for the caller
  }

  SECTION("a guess spanning both sectors is split and re-diagonalized") {
    auto p = make_parity_model({});
    const auto d_ee = macis::canonical_hf_determinant<NB>(3, 3);
    auto d_oo = d_ee;
    d_oo.reset(2);  // alpha: bath 2 (band 0) -> bath 3 (band 1)
    d_oo.set(3);
    REQUIRE(key_of(p, d_oo) == KEY_OO);
    std::vector<macis::wfn_t<NB>> dets = {d_ee, d_oo};
    std::vector<double> C = {std::sqrt(0.5), std::sqrt(0.5)};
    write_wfn_root("mixed.wfn", p.n_active, dets, C);
    p.asci_wfn_fname = "mixed.wfn";
    p.compute_asci_E0 = false;
    p.asci_E0 = 0.0;  // not used: each slice gets its own E0
    CoutCapture cap;
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    CHECK(E == Approx(E33_oo).margin(1e-8));
    CHECK(std::filesystem::exists("mixed.wfn.par_ee"));
    CHECK(std::filesystem::exists("mixed.wfn.par_oo"));
    CHECK(cap.str().find("start=guess_slice") != std::string::npos);
  }
}

TEST_CASE("Parity sectors with SYMMETRIZE_DETS, degenerate bands") {
  ScratchDir scratch("sym");
  // Band swap: 0 <-> 1, 2 <-> 3, 4 <-> 5
  auto with_swap = [](params_t& p) {
    p.asci_settings.symmetrize_dets = true;
    p.asci_settings.sym_group =
        std::make_shared<std::vector<std::vector<uint32_t>>>(
            std::vector<std::vector<uint32_t>>{{1, 0, 3, 2, 5, 4}});
  };

  SECTION("stabilizer: band swap keeps (e,e) and (o,o), not (o,e)") {
    auto p = make_parity_model({});
    const std::vector<std::vector<uint32_t>> G = {{0, 1, 2, 3, 4, 5},
                                                  {1, 0, 3, 2, 5, 4}};
    CHECK(macis::parity_stabilizer(G, *p.parity_labels, KEY_EE).size() == 2);
    CHECK(macis::parity_stabilizer(G, *p.parity_labels, KEY_OO).size() == 2);
    CHECK(macis::parity_stabilizer(G, *p.parity_labels, KEY_OE).size() == 1);
  }

  SECTION("odd N: the partner sectors tie, and the tie is reported") {
    ModelOpts o;
    o.nb = 2;
    o.degenerate = true;
    auto p = make_parity_model(o);
    with_swap(p);
    CoutCapture cap;
    const double E = macis::SolveImpurityASCI_rot<NB>(p);
    const auto out = cap.str();
    CHECK(out.find("PARITY_TIE") != std::string::npos);
    CHECK(out.find("PARITY_COVERAGE COMPLETE") != std::string::npos);

    // Same energy without symmetrization, and the same in each sector alone
    auto q = make_parity_model(o);
    CHECK(macis::SolveImpurityASCI_rot<NB>(q) == Approx(E).margin(1e-8));
    for(std::vector<int> only : {std::vector<int>{1, 0}, {0, 1}}) {
      auto r = make_parity_model(o);
      with_swap(r);
      r.parity_only = only;
      CHECK(macis::SolveImpurityASCI_rot<NB>(r) == Approx(E).margin(1e-8));
    }
  }

  SECTION("even N: symmetrized and plain parity solves agree") {
    ModelOpts o;
    o.degenerate = true;
    auto p = make_parity_model(o);
    with_swap(p);
    const double E_sym = macis::SolveImpurityASCI_rot<NB>(p);
    auto q = make_parity_model(o);
    CHECK(macis::SolveImpurityASCI_rot<NB>(q) == Approx(E_sym).margin(1e-8));
  }
}

TEST_CASE("Parity sectors inside the charge-sector search") {
  // Exact ground sector over N at eps_d = x, with every parity sector
  // solved (the parity wrapper around ED)
  auto exact = [](double x, size_t N) {
    ModelOpts o;
    o.na = (N + 1) / 2;
    o.nb = N / 2;
    o.ci = CIExpansion::CAS;
    o.eps_d = x;
    auto p = make_parity_model(o);
    const double E = macis::SolveImpurityED<NB>(p);
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    return std::make_pair(2.0 * s / p.n_imp, E);
  };
  const double x_true = -3.0;
  size_t N_true = 0;
  double Emin = 1e300;
  for(size_t N = 1; N <= 11; ++N) {
    const double E = exact(x_true, N).second;
    if(E < Emin) Emin = E, N_true = N;
  }
  const double n_true = exact(x_true, N_true).first;

  macis::ChargeSectorSettings cs;
  cs.workdir = "charge_sectors";
  for(auto ci : {CIExpansion::CAS, CIExpansion::ASCI}) {
    ScratchDir scratch(ci == CIExpansion::CAS ? "cs_cas" : "cs_asci");
    ModelOpts o;
    o.ci = ci;
    auto p = make_parity_model(o);
    p.nel_target = n_true;
    double init_mu = -3.0;
    const double mu =
        macis::Fix_Mu_sectors<NB>("brent", false, init_mu, &p, cs);
    CHECK(p.nalpha + p.nbeta == N_true);
    CHECK(mu == Approx(x_true).margin(1e-4));
  }
}

// ---- Phase 2: NROTS > 0 (parity-sector-solve-simple.md, §9) ----------------

namespace {

// Does (pq|rs) put an odd number of indices into some group?
bool breaks_parity(const macis::ParityLabels& L, size_t p, size_t q, size_t r,
                   size_t s) {
  for(size_t g = 0; g < L.ngroups; ++g) {
    const int c = (L.group_of[p] == int(g)) + (L.group_of[q] == int(g)) +
                  (L.group_of[r] == int(g)) + (L.group_of[s] == int(g));
    if(c % 2) return true;
  }
  return false;
}

}  // namespace

TEST_CASE("Per-band natural orbitals (rotate_hamiltonian_ordm_imp_bath)") {
  ScratchDir scratch("rot");
  // Degenerate bands: a band-symmetric state has every occupation twice,
  // once per band, which is where the plain imp/bath diagonalization may mix
  ModelOpts o;
  o.degenerate = GENERATE(true, false);
  auto p = make_parity_model(o);
  p.parity_only = {1, 1};
  macis::SolveImpurityASCI_rot<NB>(p);
  const size_t n = p.n_active;
  const auto& L = *p.parity_labels;

  macis::SDBuildHamiltonianGenerator<NB> H0(
      macis::matrix_span<double>(p.T_active.data(), n, n),
      macis::rank4_span<double>(p.V_active.data(), n, n, n, n));
  std::vector<double> ordm(n * n), trdm(n * n * n * n);
  H0.form_rdms(p.dets.begin(), p.dets.end(), p.dets.begin(), p.dets.end(),
               p.C.data(), macis::matrix_span<double>(ordm.data(), n, n),
               macis::rank4_span<double>(trdm.data(), n, n, n, n));
  // A parity-pure state has no cross-band 1-RDM element at all
  CHECK(macis::max_off_group(ordm.data(), L) == 0.0);

  auto T = p.T_active, V = p.V_active;
  macis::SDBuildHamiltonianGenerator<NB> H(
      macis::matrix_span<double>(T.data(), n, n),
      macis::rank4_span<double>(V.data(), n, n, n, n));
  std::vector<double> U(n * n), occ(n);
  H.rotate_hamiltonian_ordm_imp_bath(ordm.data(), p.n_imp, U.data(), false,
                                     occ.data(), &L.group_of);

  // Every orbital keeps its group, exactly
  CHECK(macis::max_off_group(U.data(), L) == 0.0);
  // U is orthogonal and diagonalizes each (impurity|bath) x group block of
  // the 1-RDM, with occ on the diagonal (impurity-bath elements remain, as
  // in the legacy imp/bath rotation), occupations descending in each block
  auto same_block = [&](size_t a, size_t b) {
    return L.group_of[a] == L.group_of[b] and (a < p.n_imp) == (b < p.n_imp);
  };
  for(size_t a = 0; a < n; ++a)
    for(size_t b = 0; b < n; ++b) {
      double uu = 0., unu = 0.;
      for(size_t i = 0; i < n; ++i) {
        uu += U[i + a * n] * U[i + b * n];
        for(size_t j = 0; j < n; ++j)
          unu += U[i + a * n] * ordm[i + j * n] * U[j + b * n];
      }
      CHECK(uu == Approx(a == b ? 1.0 : 0.0).margin(1e-12));
      if(same_block(a, b))
        CHECK(unu == Approx(a == b ? occ[a] : 0.0).margin(1e-10));
      if(same_block(a, b) and a < b) CHECK(occ[a] >= occ[b]);
    }
  // The rotated integrals still conserve every band parity, exactly
  for(size_t a = 0; a < n; ++a)
    for(size_t b = 0; b < n; ++b)
      if(L.group_of[a] != L.group_of[b]) CHECK(T[a + b * n] == 0.0);
  size_t nbroken = 0;
  for(size_t a = 0; a < n; ++a)
    for(size_t b = 0; b < n; ++b)
      for(size_t c = 0; c < n; ++c)
        for(size_t d = 0; d < n; ++d)
          if(breaks_parity(L, a, b, c, d) and
             V[a + b * n + c * n * n + d * n * n * n] != 0.0)
            ++nbroken;
  CHECK(nbroken == 0);

  // Without group_of: the legacy imp/bath blocks. They sort the natural
  // orbitals by occupation across bands, so an orbital index changes band
  // whenever the bands' occupations are not in band order (here, impurity
  // band 1 is fuller than band 0): the labels break even without degeneracy.
  // With degenerate bands gesvd may in addition mix the pairs (LAPACK-
  // dependent, so only recorded).
  auto T2 = p.T_active, V2 = p.V_active;
  macis::SDBuildHamiltonianGenerator<NB> H2(
      macis::matrix_span<double>(T2.data(), n, n),
      macis::rank4_span<double>(V2.data(), n, n, n, n));
  std::vector<double> U2(n * n), occ2(n);
  H2.rotate_hamiltonian_ordm_imp_bath(ordm.data(), p.n_imp, U2.data(), false,
                                      occ2.data());
  INFO("legacy imp/bath rotation, largest cross-band element = "
       << macis::max_off_group(U2.data(), L));
  if(!o.degenerate) CHECK(macis::max_off_group(U2.data(), L) > 0.5);
  // Same spectrum, only arranged differently
  auto s1 = occ, s2 = occ2;
  std::sort(s1.begin(), s1.end());
  std::sort(s2.begin(), s2.end());
  for(size_t a = 0; a < n; ++a) CHECK(s1[a] == Approx(s2[a]).margin(1e-12));
}

TEST_CASE("Parity sectors with NROTS > 0") {
  ScratchDir scratch("nrots");
  struct Case {
    size_t na, nb;
    std::vector<int> only;
    double E;
  };
  const std::vector<Case> cases = {{3, 3, {0, 0}, E33_ee},
                                   {3, 3, {1, 1}, E33_oo},
                                   {3, 2, {0, 1}, E32_eo},
                                   {3, 2, {1, 0}, E32_oe}};

  SECTION("each sector against exact diagonalization, NROTS = 2") {
    for(bool degenerate : {false, true}) {
      for(const auto& c : cases) {
        ModelOpts o;
        o.na = c.na;
        o.nb = c.nb;
        o.degenerate = degenerate;
        auto p = make_parity_model(o);
        p.asci_settings.nrots = 2;
        p.parity_only = c.only;
        SearchLogCapture search_log;
        CoutCapture cap;
        const double E = macis::SolveImpurityASCI_rot<NB>(p);
        const auto out = cap.str();
        const auto slog = search_log.str();
        if(!degenerate) CHECK(E == Approx(c.E).margin(1e-8));
        // Same energy as NROTS = 0 in the same sector
        auto q = make_parity_model(o);
        q.parity_only = c.only;
        CHECK(E == Approx(macis::SolveImpurityASCI_rot<NB>(q)).margin(1e-8));
        CHECK(out.find("PARITY_COVERAGE COMPLETE") != std::string::npos);
        // Two rotations, a restart in the sector after each
        size_t nrestart = 0;
        for(auto pos = out.find("PARITY_SOLVE: restart for sector");
            pos != std::string::npos;
            pos = out.find("PARITY_SOLVE: restart for sector", pos + 1))
          ++nrestart;
        CHECK(nrestart == 2);
        CHECK(slog.find("dropped 0 candidates") != std::string::npos);
        CHECK(slog.find("dropped 1") == std::string::npos);
        // The accumulated rotation keeps every orbital in its band
        CHECK(macis::max_off_group(p.orb_rot.data(), *p.parity_labels) == 0.0);
        const uint32_t want = uint32_t(c.only[0]) | uint32_t(c.only[1]) << 1;
        for(const auto& d : p.dets) CHECK(key_of(p, d) == want);
      }
    }
  }

  SECTION("full solve: the wrapper returns the ground sector") {
    for(size_t nb : {size_t(3), size_t(2)}) {
      ModelOpts o;
      o.nb = nb;
      auto p = make_parity_model(o);
      p.asci_settings.nrots = 2;
      CoutCapture cap;
      const double E = macis::SolveImpurityASCI_rot<NB>(p);
      CHECK(E == Approx(nb == 3 ? E33_oo : E32_oe).margin(1e-8));
      CHECK(cap.str().find("PARITY_COVERAGE COMPLETE") != std::string::npos);
      // occs are in the original orbital basis: the band counts add up
      double nel = 0.;
      for(auto x : p.occs) nel += 2 * x;
      CHECK(nel == Approx(3.0 + nb).margin(1e-8));
    }
  }

  SECTION("a guess file is still refused with NROTS > 0") {
    auto ref = make_parity_model({});
    ref.parity_only = {1, 1};
    const double E_ref = macis::SolveImpurityASCI_rot<NB>(ref);
    write_wfn_root("oo.wfn", ref.n_active, ref.dets, ref.C);
    auto p = make_parity_model({});
    p.asci_settings.nrots = 2;
    p.asci_wfn_fname = "oo.wfn";
    p.compute_asci_E0 = false;
    p.asci_E0 = E_ref;
    p.parity_only = {1, 1};
    CHECK_THROWS_WITH(macis::SolveImpurityASCI_rot<NB>(p),
                      Catch::Contains("NROTS > 0 is not supported"));
  }
}

TEST_CASE("Inherited basis under PARITY_SOLVE") {
  auto p = make_parity_model({});
  const size_t n = p.n_active;
  macis::charge_sectors::basis_t U(n * n, 0.0);
  for(size_t i = 0; i < n; ++i) U[i + i * n] = 1.0;
  // Per-band: a rotation between bath orbitals 2 and 4 (both band 0)
  const double c = std::cos(0.3), s = std::sin(0.3);
  U[2 + 2 * n] = c, U[4 + 2 * n] = s, U[2 + 4 * n] = -s, U[4 + 4 * n] = c;
  CHECK_NOTHROW(macis::charge_sectors::rotate_active<NB>(p, U));

  // Across bands: bath orbitals 2 (band 0) and 3 (band 1)
  auto q = make_parity_model({});
  macis::charge_sectors::basis_t W(n * n, 0.0);
  for(size_t i = 0; i < n; ++i) W[i + i * n] = 1.0;
  W[2 + 2 * n] = c, W[3 + 2 * n] = s, W[2 + 3 * n] = -s, W[3 + 3 * n] = c;
  CHECK_THROWS_WITH(macis::charge_sectors::rotate_active<NB>(q, W),
                    Catch::Contains("mixes band groups"));
}

TEST_CASE("Parity sectors inside the charge-sector search at NROTS = 2") {
  // Same target as the NROTS = 0 test above: the exact ground sector at
  // eps_d = -3 over N, from the parity wrapper around ED
  auto exact = [](size_t N) {
    ModelOpts o;
    o.na = (N + 1) / 2;
    o.nb = N / 2;
    o.ci = CIExpansion::CAS;
    auto p = make_parity_model(o);
    const double E = macis::SolveImpurityED<NB>(p);
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    return std::make_pair(2.0 * s / p.n_imp, E);
  };
  size_t N_true = 0;
  double Emin = 1e300;
  for(size_t N = 1; N <= 11; ++N) {
    const double E = exact(N).second;
    if(E < Emin) Emin = E, N_true = N;
  }

  ScratchDir scratch("cs_nrots");
  macis::ChargeSectorSettings cs;
  cs.workdir = "charge_sectors";
  auto p = make_parity_model({});
  p.asci_settings.nrots = 2;
  p.nel_target = exact(N_true).first;
  double init_mu = -3.0;
  const double mu = macis::Fix_Mu_sectors<NB>("brent", false, init_mu, &p, cs);
  CHECK(p.nalpha + p.nbeta == N_true);
  CHECK(mu == Approx(-3.0).margin(1e-4));
  CHECK(p.E == Approx(Emin).margin(1e-6));
}
