#include <cmath>
#include <filesystem>
#include <macis/doping/charge_sectors.hpp>
#include <macis/doping/fix_mu.hpp>

#include "ut_common.hpp"

namespace {

constexpr size_t NB = 64;
using params_t = macis::impurity_params<NB>;

// Two-band Kanamori impurity (orbitals 0, 1) with four bath sites: each bath
// level couples to its own band and, more weakly, to the other one, which
// breaks the per-band parity conservation. The impurity diagonal (-mu) is left
// to the mu search.
params_t make_model(CIExpansion ci_exp, size_t na, size_t nb) {
  const size_t n_imp = 2, n = 6;
  const double U = 6.0, J = 0.8;
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
  p.ci_exp = ci_exp;
  p.T.assign(n * n, 0.0);
  p.Td.assign(n * n, 0.0);
  p.V.assign(n * n * n * n, 0.0);

  const double eps[4] = {-2.0, -0.7, 0.6, 1.9};
  for(size_t k = 0; k < 4; ++k) {
    const size_t b = n_imp + k, band = k % 2, other = 1 - band;
    p.T[b + b * n] = eps[k];
    p.T[band + b * n] = p.T[b + band * n] = 0.5;
    p.T[other + b * n] = p.T[b + other * n] = 0.15;
  }
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
  p.asci_settings.ntdets_max = 1000;  // the largest sector has 400 determinants
  p.asci_settings.ntdets_min = 10;
  p.asci_settings.ncdets_max = 400;
  p.asci_settings.max_refine_iter = 6;
  p.asci_settings.refine_energy_tol = 1e-8;
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

  // doping
  p.dstep = 2e-2;
  p.abs_tol = 1e-7;
  p.maxiter = 100;
  p.print_doping = false;
  p.init_shift = 2.0;
  p.delta_CFS = 0.0;
  p.cheap_mode = false;
  return p;
}

// Filling (electrons per impurity orbital) and energy of the lowest state of
// the sector with N electrons at impurity level x, by ED.
std::pair<double, double> exact_sector(double x, size_t N) {
  auto p = make_model(CIExpansion::CAS, (N + 1) / 2, N / 2);
  for(size_t i = 0; i < p.n_imp; ++i) p.T[i * p.norb + i] = x;
  macis::charge_sectors::rebuild_active<NB>(p);
  const double E = macis::SolveImpurityED<NB>(p);
  double s = 0;
  for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
  return {2.0 * s / p.n_imp, E};
}

// The ground-state sector at x, over every N of the 6-orbital problem.
size_t exact_ground_sector(double x) {
  size_t best = 0;
  double Emin = 1e300;
  for(size_t N = 1; N <= 11; ++N) {
    const double E = exact_sector(x, N).second;
    if(E < Emin) Emin = E, best = N;
  }
  return best;
}

// The sector search writes seed files and GS_charge_sector.dat: keep them out
// of the working directory of the test run.
struct ScratchDir {
  std::filesystem::path old = std::filesystem::current_path();
  std::filesystem::path dir;
  ScratchDir(const std::string& tag) {
    dir = std::filesystem::temp_directory_path() / ("macis_charge_sectors_" + tag);
    std::filesystem::create_directories(dir);
    std::filesystem::current_path(dir);
  }
  ~ScratchDir() { std::filesystem::current_path(old); }
};

}  // namespace

TEST_CASE("Charge sector search in the doping mu search") {
  // At x = -3 the ground state is the N = 5 sector, with a filling of about
  // 0.585 electrons per orbital, and N = 6 lies 0.2 above it.
  const double x_true = -3.0;
  const size_t N_true = exact_ground_sector(x_true);
  const double n_true = exact_sector(x_true, N_true).first;
  REQUIRE(N_true == 5);

  macis::ChargeSectorSettings cs;
  cs.workdir = "charge_sectors";

  auto run = [&](CIExpansion ci, size_t na, size_t nb, double target, size_t nrots = 0) {
    params_t p = make_model(ci, na, nb);
    p.asci_settings.nrots = nrots;
    p.nel_target = target;
    double init_mu = -3.0;
    const double mu = macis::Fix_Mu_sectors<NB>("brent", false, init_mu, &p, cs);
    return std::make_pair(mu, p);
  };

  SECTION("starting one sector above the ground state, CAS") {
    ScratchDir scratch("cas6");
    auto [mu, p] = run(CIExpansion::CAS, 3, 3, n_true);
    CHECK(p.nalpha + p.nbeta == N_true);
    CHECK(mu == Approx(x_true).margin(1e-5));
    double s = 0;
    for(size_t i = 0; i < p.n_imp; ++i) s += p.occs[i];
    CHECK(2.0 * s / p.n_imp == Approx(n_true).margin(1e-6));
    CHECK(std::filesystem::exists("GS_charge_sector.dat"));
  }

  SECTION("starting one sector below the ground state, CAS") {
    ScratchDir scratch("cas4");
    auto [mu, p] = run(CIExpansion::CAS, 2, 2, n_true);
    CHECK(p.nalpha + p.nbeta == N_true);
    CHECK(mu == Approx(x_true).margin(1e-5));
  }

  SECTION("starting in the ground-state sector does not switch") {
    ScratchDir scratch("cas5");
    auto [mu, p] = run(CIExpansion::CAS, 3, 2, n_true);
    CHECK(p.nalpha + p.nbeta == N_true);
    CHECK(mu == Approx(x_true).margin(1e-5));
  }

  SECTION("ASCI with warm-started neighbours, NROTS = 0 and 2") {
    for(size_t nrots : {size_t(0), size_t(2)}) {
      ScratchDir scratch("asci" + std::to_string(nrots));
      auto [mu, p] = run(CIExpansion::ASCI, 3, 3, n_true, nrots);
      CHECK(p.nalpha + p.nbeta == N_true);
      CHECK(mu == Approx(x_true).margin(1e-4));
    }
  }

  SECTION("a target inside a jump of the ground-state filling is an error") {
    // 0.75 electrons per orbital is reached by N = 5 at x = -4.12 and by
    // N = 6 at x = -3.03, but the other one is the ground state at each of
    // them: no ground state has this filling.
    const double target = 0.75;
    REQUIRE(exact_ground_sector(-3.030) == 5);
    REQUIRE(exact_ground_sector(-4.117) == 6);
    ScratchDir scratch("jump");
    CHECK_THROWS_WITH(run(CIExpansion::CAS, 3, 3, target),
                      Catch::Contains("lies in a jump"));
  }

  SECTION("a sector search that is switched off keeps the input sector") {
    ScratchDir scratch("off");
    cs.enabled = false;
    auto [mu, p] = run(CIExpansion::CAS, 3, 3, n_true);
    CHECK(p.nalpha + p.nbeta == 6);
    cs.enabled = true;
  }
}
