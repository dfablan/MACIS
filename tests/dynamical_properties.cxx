/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <map>
#include <macis/csr_hamiltonian.hpp>
#include <macis/gf/dynamical_properties.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <utility>

#include "ut_common.hpp"

namespace {

constexpr size_t N = 64;  // 32 spatial orbitals max; we use 4.

// Build a determinant from explicit alpha / beta spatial-orbital occupation
// lists. Alpha orbital i -> bit i, beta orbital i -> bit i + N/2.
macis::wfn_t<N> make_det(const std::vector<int>& alpha_occ,
                         const std::vector<int>& beta_occ) {
  macis::wfn_t<N> det = 0;
  for(int o : alpha_occ) det.set(o);
  for(int o : beta_occ) det.set(o + N / 2);
  return det;
}

// Dense Hamiltonian over `dets` (replicated CSR -> dense Eigen matrix), built
// with the same generator used by RunResolventSz.
template <class Gen>
Eigen::MatrixXd dense_hamiltonian(std::vector<macis::wfn_t<N>>& dets,
                                  Gen& ham_gen) {
  auto H = macis::make_csr_hamiltonian_block<int32_t>(
      dets.begin(), dets.end(), dets.begin(), dets.end(), ham_gen, 1e-16);
  const int n = int(dets.size());
  Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(n, n);
  const auto& rowptr = H.rowptr();
  const auto& colind = H.colind();
  const auto& nzval = H.nzval();
  for(int i = 0; i < n; ++i)
    for(int p = rowptr[i]; p < rowptr[i + 1]; ++p) Hd(i, colind[p]) = nzval[p];
  return Hd;
}

// One- and two-body integrals of a 4-orbital model with distinct on-site
// energies, uniform hopping and on-site U, chosen to have no spatial symmetry.
void orbital_matrix_integrals(std::vector<double>& T, std::vector<double>& V) {
  const size_t n = 4;
  T.assign(n * n, 0.0);
  V.assign(n * n * n * n, 0.0);
  for(size_t p = 0; p < n; ++p) {
    T[p * n + p] = -1.0 - 0.1 * double(p);
    for(size_t q = 0; q < n; ++q)
      if(p != q) T[p * n + q] = -0.25;
    V[((p * n + p) * n + p) * n + p] = 1.0;
  }
}

// Complete (2 alpha, 2 beta) FCI space over 4 orbitals: 36 determinants.
std::vector<macis::wfn_t<N>> half_filled_fci_dets() {
  std::vector<macis::wfn_t<N>> dets;
  for(int a0 = 0; a0 < 4; ++a0)
    for(int a1 = a0 + 1; a1 < 4; ++a1)
      for(int b0 = 0; b0 < 4; ++b0)
        for(int b1 = b0 + 1; b1 < 4; ++b1)
          dets.push_back(make_det({a0, a1}, {b0, b1}));
  return dets;
}

// All n_imp^2 seeds S_{mu nu}|psi0> as columns, pair index mu * n_imp + nu.
Eigen::MatrixXd spin_bilinear_seeds(const Eigen::VectorXd& psi0,
                                    const std::vector<macis::wfn_t<N>>& dets,
                                    size_t n_imp) {
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);
  Eigen::MatrixXd seeds(dets.size(), n_imp * n_imp);
  for(size_t mu = 0; mu < n_imp; ++mu)
    for(size_t nu = 0; nu < n_imp; ++nu)
      seeds.col(mu * n_imp + nu) =
          macis::apply_spin_bilinear<N>(psi0, dets, index, mu, nu);
  return seeds;
}

// Exact Lehmann matrix sum_n <phi_k|n><n|phi_l> / (z - (E_n - E0)).
std::complex<double> lehmann_element(
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& es,
    const Eigen::MatrixXd& overlaps, size_t k, size_t l, double E0,
    std::complex<double> z) {
  std::complex<double> value(0.0, 0.0);
  for(Eigen::Index n = 0; n < es.eigenvalues().size(); ++n)
    value += overlaps(n, k) * overlaps(n, l) / (z - (es.eigenvalues()(n) - E0));
  return value;
}

}  // namespace

TEST_CASE("Dynamical properties - sz_imp_value and apply_diagonal_operator") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t n_imp = 2;
  const size_t n_active = 4;

  // det A: impurity = (up:0, dn:1)  -> n_imp_up=1, n_imp_dn=1 -> Sz_imp = 0
  //        (bath orbital 2 up, 3 dn keeps total Sz = 0)
  auto detA = make_det(/*alpha*/ {0, 2}, /*beta*/ {1, 3});
  // det B: impurity = (up:0,1 ; dn: none) -> n_imp_up=2, n_imp_dn=0 ->
  // Sz_imp=+1
  //        bath: (up: none ; dn: 2,3) so total Sz = 0
  auto detB = make_det(/*alpha*/ {0, 1}, /*beta*/ {2, 3});
  // det C: impurity = (up: none ; dn:0,1) -> n_imp_up=0, n_imp_dn=2 ->
  // Sz_imp=-1
  auto detC = make_det(/*alpha*/ {2, 3}, /*beta*/ {0, 1});

  SECTION("sz_imp_value matches hand-computed 0.5*(n_up_imp - n_dn_imp)") {
    REQUIRE(macis::sz_imp_value<N>(detA, n_imp, n_active) ==
            Approx(0.0).margin(1e-12));
    REQUIRE(macis::sz_imp_value<N>(detB, n_imp, n_active) ==
            Approx(1.0).epsilon(1e-12));
    REQUIRE(macis::sz_imp_value<N>(detC, n_imp, n_active) ==
            Approx(-1.0).epsilon(1e-12));
  }

  SECTION("apply_diagonal_operator scales each coefficient by the scalar fn") {
    std::vector<macis::wfn_t<N>> dets = {detA, detB, detC};
    Eigen::VectorXd c(3);
    c << 0.3, -0.7, 0.5;

    Eigen::VectorXd v = macis::apply_diagonal_operator<N>(
        c, dets, [&](const macis::wfn_t<N>& d) {
          return macis::sz_imp_value<N>(d, n_imp, n_active);
        });

    REQUIRE(v(0) == Approx(0.3 * 0.0).margin(1e-12));    // Sz_imp(A)=0
    REQUIRE(v(1) == Approx(-0.7 * 1.0).epsilon(1e-12));  // Sz_imp(B)=+1
    REQUIRE(v(2) == Approx(0.5 * -1.0).epsilon(1e-12));  // Sz_imp(C)=-1
  }
}

TEST_CASE("Dynamical properties - RunResolventSz vs exact Lehmann sum") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t norb = n_active;

  // Synthetic, symmetric one- and two-body integrals (Hermitian H). The exact
  // values do not matter; we only need a nontrivial Hamiltonian that couples
  // the determinants so that Sz_imp|psi0> has real dynamical structure.
  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb * norb * norb * norb, 0.0);
  for(size_t p = 0; p < norb; ++p) {
    T[p * norb + p] = -1.0 - 0.1 * double(p);  // on-site energies
    for(size_t q = 0; q < norb; ++q)
      if(p != q) T[p * norb + q] = -0.25;  // hopping
  }
  for(size_t p = 0; p < norb; ++p)
    V[((p * norb + p) * norb + p) * norb + p] = 1.0;  // Hubbard-like U

  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  // Full Sz_tot = 0 sector (2 alpha + 2 beta electrons in 4 orbitals): every
  // determinant has total Sz = 0, as requested.
  std::vector<macis::wfn_t<N>> dets;
  std::vector<int> orbs = {0, 1, 2, 3};
  for(size_t a0 = 0; a0 < orbs.size(); ++a0)
    for(size_t a1 = a0 + 1; a1 < orbs.size(); ++a1)
      for(size_t b0 = 0; b0 < orbs.size(); ++b0)
        for(size_t b1 = b0 + 1; b1 < orbs.size(); ++b1)
          dets.push_back(make_det({orbs[a0], orbs[a1]}, {orbs[b0], orbs[b1]}));
  const int ndet = int(dets.size());  // C(4,2)^2 = 36

  // Dense H + ground state of this sector.
  Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  // Frequency grid: real axis with finite broadening eta.
  macis::GFSettings settings;
  settings.nLanIts = 200;
  settings.saveGFmats = false;
  const double eta = 0.2;
  std::vector<std::complex<double>> ws;
  for(double w = -6.0; w <= 6.0 + 1e-9; w += 1.0) ws.emplace_back(w, eta);

  auto R = macis::RunResolventSz<N, int32_t>(psi0, ham_gen, dets, n_imp,
                                             n_active, E0, ws, settings);
  REQUIRE(R.size() == ws.size());

  // Exact Lehmann reference:
  //   v = Sz_imp |psi0>,
  //   R_exact(w) = sum_n |<n|v>|^2 / (w - (E_n - E0))   (ispart=true sign).
  Eigen::VectorXd v(ndet);
  for(int k = 0; k < ndet; ++k)
    v(k) = psi0(k) * macis::sz_imp_value<N>(dets[k], n_imp, n_active);

  // v must be nonzero: with n_imp < n_active, Sz_imp|psi0> has real structure
  // even though every determinant (and psi0) has total Sz = 0.
  const double vnorm2 = v.squaredNorm();
  REQUIRE(vnorm2 > 1e-8);

  Eigen::VectorXd overlaps = es.eigenvectors().transpose() * v;  // <n|v>
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    std::complex<double> ref(0.0, 0.0);
    for(int n = 0; n < ndet; ++n) {
      const double wn = es.eigenvalues()(n) - E0;  // excitation energy >= 0
      ref += (overlaps(n) * overlaps(n)) / (ws[iw] - wn);
    }
    REQUIRE(std::real(R[iw]) ==
            Approx(std::real(ref)).epsilon(1e-6).margin(1e-8));
    REQUIRE(std::imag(R[iw]) ==
            Approx(std::imag(ref)).epsilon(1e-6).margin(1e-8));
  }

  SECTION("first spectral moment is non-negative (excitations above GS)") {
    // m1 = <v|(H - E0)|v> = sum_n |<n|v>|^2 (E_n - E0) >= 0.
    double m1 = 0.0;
    for(int n = 0; n < ndet; ++n)
      m1 += (overlaps(n) * overlaps(n)) * (es.eigenvalues()(n) - E0);
    REQUIRE(m1 >= -1e-10);
  }

  SECTION("retarded spectral function is non-negative: Im R(w) <= 0") {
    for(size_t iw = 0; iw < ws.size(); ++iw) REQUIRE(std::imag(R[iw]) <= 1e-8);
  }
}

TEST_CASE(
    "Dynamical properties - all-active impurity gives zero response for an "
    "Sz_tot = 0 state") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // When n_imp == n_active, Sz_imp == Sz_tot. For a state in which every
  // determinant has total Sz = 0, Sz_imp|psi0> == 0 exactly, so R(w) == 0.
  const size_t n_active = 4;
  const size_t n_imp = n_active;  // all-active impurity
  const size_t norb = n_active;

  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb * norb * norb * norb, 0.0);
  for(size_t p = 0; p < norb; ++p) {
    T[p * norb + p] = -1.0 - 0.1 * double(p);
    for(size_t q = 0; q < norb; ++q)
      if(p != q) T[p * norb + q] = -0.25;
  }
  for(size_t p = 0; p < norb; ++p)
    V[((p * norb + p) * norb + p) * norb + p] = 1.0;

  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  std::vector<macis::wfn_t<N>> dets;
  std::vector<int> orbs = {0, 1, 2, 3};
  for(size_t a0 = 0; a0 < orbs.size(); ++a0)
    for(size_t a1 = a0 + 1; a1 < orbs.size(); ++a1)
      for(size_t b0 = 0; b0 < orbs.size(); ++b0)
        for(size_t b1 = b0 + 1; b1 < orbs.size(); ++b1)
          dets.push_back(make_det({orbs[a0], orbs[a1]}, {orbs[b0], orbs[b1]}));

  Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  // Sz_imp|psi0> should vanish identically here.
  Eigen::VectorXd v(int(dets.size()));
  for(int k = 0; k < int(dets.size()); ++k)
    v(k) = psi0(k) * macis::sz_imp_value<N>(dets[k], n_imp, n_active);
  REQUIRE(v.squaredNorm() == Approx(0.0).margin(1e-12));

  macis::GFSettings settings;
  settings.nLanIts = 100;
  std::vector<std::complex<double>> ws = {{0.5, 0.1}, {1.0, 0.1}, {2.0, 0.1}};

  auto R = macis::RunResolventSz<N, int32_t>(psi0, ham_gen, dets, n_imp,
                                             n_active, E0, ws, settings);
  for(const auto& r : R) {
    REQUIRE(std::real(r) == Approx(0.0).margin(1e-10));
    REQUIRE(std::imag(r) == Approx(0.0).margin(1e-10));
  }
}

TEST_CASE("Dynamical properties - weighted_imp_value orbital-to-bit map") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // decompose_det packs impurity occupations in REVERSE (impurity_rdm.hpp:
  // "if(alpha[n_imp - 1 - p]) out.imp_up |= (1ULL << p);"). weighted_imp_value
  // must un-reverse that mapping so that w[i] is applied to impurity orbital
  // i as laid out by the caller (make_det: alpha orbital i -> bit i), not to
  // whatever bit position decompose_det happens to store it at. A distinct
  // weight per orbital (rather than a uniform one) is what catches the
  // reversal: getting it backwards would silently swap w[i] <-> w[n_imp-1-i].
  const size_t n_imp = 3;
  const size_t n_active = 5;
  const std::vector<double> w = {2.0, 5.0, -3.0};

  for(size_t i = 0; i < n_imp; ++i) {
    // Single spin-up electron in impurity orbital i, nothing else occupied.
    auto det = make_det(/*alpha*/ {int(i)}, /*beta*/ {});
    const double val = macis::weighted_imp_value<N>(
        det, w, macis::DiagChannel::Charge, n_imp, n_active);
    REQUIRE(val == Approx(w[i]).epsilon(1e-12));
  }
}

TEST_CASE(
    "Dynamical properties - sz_imp_value matches weighted_imp_value with "
    "uniform spin weights") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t n_imp = 2;
  const size_t n_active = 4;
  const auto w = macis::make_uniform_spin_weights(/*nbands=*/2, /*nsites=*/1);
  REQUIRE(w.size() == n_imp);

  auto detA = make_det(/*alpha*/ {0, 2}, /*beta*/ {1, 3});
  auto detB = make_det(/*alpha*/ {0, 1}, /*beta*/ {2, 3});
  auto detC = make_det(/*alpha*/ {2, 3}, /*beta*/ {0, 1});

  for(auto [det, expected] :
      {std::pair{detA, 0.0}, std::pair{detB, 1.0}, std::pair{detC, -1.0}}) {
    const double via_weighted = macis::weighted_imp_value<N>(
        det, w, macis::DiagChannel::Spin, n_imp, n_active);
    const double via_sz = macis::sz_imp_value<N>(det, n_imp, n_active);
    REQUIRE(via_weighted == Approx(expected).margin(1e-12));
    REQUIRE(via_weighted == Approx(via_sz).margin(1e-12));
  }
}

TEST_CASE("Dynamical properties - weight builders are traceless / validate") {
  ROOT_ONLY(MPI_COMM_WORLD);

  auto sum = [](const std::vector<double>& v) {
    double s = 0.0;
    for(double x : v) s += x;
    return s;
  };

  SECTION("orbital Cartan generators sum to zero") {
    REQUIRE(sum(macis::make_orbital_cartan_weights(2, 1, 3)) ==
            Approx(0.0).margin(1e-12));
    REQUIRE(sum(macis::make_orbital_cartan_weights(3, 1, 3)) ==
            Approx(0.0).margin(1e-12));
    REQUIRE(sum(macis::make_orbital_cartan_weights(3, 1, 8)) ==
            Approx(0.0).margin(1e-12));
    // Two-site cluster: still band-uniform across sites, still traceless.
    REQUIRE(sum(macis::make_orbital_cartan_weights(3, 2, 8)) ==
            Approx(0.0).margin(1e-12));
  }

  SECTION("T^3 vs T^8 agree in magnitude structure for 3 degenerate bands") {
    const auto w3 = macis::make_orbital_cartan_weights(3, 1, 3);
    const auto w8 = macis::make_orbital_cartan_weights(3, 1, 8);
    REQUIRE(w3 == std::vector<double>{0.5, -0.5, 0.0});
    const double c = 1.0 / (2.0 * std::sqrt(3.0));
    REQUIRE(w8[0] == Approx(c).epsilon(1e-12));
    REQUIRE(w8[1] == Approx(c).epsilon(1e-12));
    REQUIRE(w8[2] == Approx(-2.0 * c).epsilon(1e-12));
  }

  SECTION("make_orbital_cartan_weights throws for nbands == 1") {
    REQUIRE_THROWS_AS(macis::make_orbital_cartan_weights(1, 1, 3),
                      std::runtime_error);
  }

  SECTION("make_orbital_cartan_weights throws for T^8 unless nbands == 3") {
    REQUIRE_THROWS_AS(macis::make_orbital_cartan_weights(2, 1, 8),
                      std::runtime_error);
  }

  SECTION("make_orbital_cartan_weights throws for an invalid which") {
    REQUIRE_THROWS_AS(macis::make_orbital_cartan_weights(3, 1, 2),
                      std::runtime_error);
  }

  SECTION("make_staggered_spin_weights requires nsites == 2") {
    REQUIRE_THROWS_AS(macis::make_staggered_spin_weights(2, 1),
                      std::runtime_error);
    const auto w = macis::make_staggered_spin_weights(2, 2);
    REQUIRE(w.size() == 4);
    REQUIRE(sum(w) == Approx(0.0).margin(1e-12));
  }

  SECTION("make_uniform_spin_weights has the expected size and values") {
    const auto w = macis::make_uniform_spin_weights(3, 2);
    REQUIRE(w.size() == 6);
    for(double x : w) REQUIRE(x == Approx(1.0).margin(1e-12));
  }
}

TEST_CASE(
    "Dynamical properties - RunResolventWeighted (orbital T^3) vs exact "
    "Lehmann sum") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Mirrors "RunResolventSz vs exact Lehmann sum" above, with the impurity
  // Sz operator replaced by the orbital isospin generator T^3 (weights
  // (+1/2, -1/2) on the two impurity bands, CHARGE channel).
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t norb = n_active;

  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb * norb * norb * norb, 0.0);
  for(size_t p = 0; p < norb; ++p) {
    T[p * norb + p] = -1.0 - 0.1 * double(p);
    for(size_t q = 0; q < norb; ++q)
      if(p != q) T[p * norb + q] = -0.25;
  }
  for(size_t p = 0; p < norb; ++p)
    V[((p * norb + p) * norb + p) * norb + p] = 1.0;

  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  std::vector<macis::wfn_t<N>> dets;
  std::vector<int> orbs = {0, 1, 2, 3};
  for(size_t a0 = 0; a0 < orbs.size(); ++a0)
    for(size_t a1 = a0 + 1; a1 < orbs.size(); ++a1)
      for(size_t b0 = 0; b0 < orbs.size(); ++b0)
        for(size_t b1 = b0 + 1; b1 < orbs.size(); ++b1)
          dets.push_back(make_det({orbs[a0], orbs[a1]}, {orbs[b0], orbs[b1]}));
  const int ndet = int(dets.size());

  Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  macis::GFSettings settings;
  settings.nLanIts = 200;
  settings.saveGFmats = false;
  const double eta = 0.2;
  std::vector<std::complex<double>> ws;
  for(double w = -6.0; w <= 6.0 + 1e-9; w += 1.0) ws.emplace_back(w, eta);

  const auto w3 = macis::make_orbital_cartan_weights(/*nbands=*/2,
                                                      /*nsites=*/1,
                                                      /*which=*/3);
  auto R = macis::RunResolventWeighted<N, int32_t>(
      psi0, ham_gen, dets, w3, macis::DiagChannel::Charge, n_imp, n_active,
      E0, ws, settings);
  REQUIRE(R.size() == ws.size());

  // Exact Lehmann reference: v = T^3 |psi0>.
  Eigen::VectorXd v(ndet);
  for(int k = 0; k < ndet; ++k)
    v(k) = psi0(k) * macis::weighted_imp_value<N>(dets[k], w3,
                                                   macis::DiagChannel::Charge,
                                                   n_imp, n_active);

  const double vnorm2 = v.squaredNorm();
  REQUIRE(vnorm2 > 1e-8);

  Eigen::VectorXd overlaps = es.eigenvectors().transpose() * v;
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    std::complex<double> ref(0.0, 0.0);
    for(int n = 0; n < ndet; ++n) {
      const double wn = es.eigenvalues()(n) - E0;
      ref += (overlaps(n) * overlaps(n)) / (ws[iw] - wn);
    }
    REQUIRE(std::real(R[iw]) ==
            Approx(std::real(ref)).epsilon(1e-6).margin(1e-8));
    REQUIRE(std::imag(R[iw]) ==
            Approx(std::imag(ref)).epsilon(1e-6).margin(1e-8));
  }
}

TEST_CASE(
    "Dynamical properties - subtract_mean cancels the elastic pole") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Same setup as the T^3 Lehmann test above, but comparing the plain
  // resolvent of O against the fluctuation resolvent of delta_O = O - <O>.
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t norb = n_active;

  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb * norb * norb * norb, 0.0);
  for(size_t p = 0; p < norb; ++p) {
    T[p * norb + p] = -1.0 - 0.1 * double(p);
    for(size_t q = 0; q < norb; ++q)
      if(p != q) T[p * norb + q] = -0.25;
  }
  for(size_t p = 0; p < norb; ++p)
    V[((p * norb + p) * norb + p) * norb + p] = 1.0;

  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  std::vector<macis::wfn_t<N>> dets;
  std::vector<int> orbs = {0, 1, 2, 3};
  for(size_t a0 = 0; a0 < orbs.size(); ++a0)
    for(size_t a1 = a0 + 1; a1 < orbs.size(); ++a1)
      for(size_t b0 = 0; b0 < orbs.size(); ++b0)
        for(size_t b1 = b0 + 1; b1 < orbs.size(); ++b1)
          dets.push_back(make_det({orbs[a0], orbs[a1]}, {orbs[b0], orbs[b1]}));
  const int ndet = int(dets.size());

  Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  macis::GFSettings settings;
  settings.nLanIts = 200;
  settings.saveGFmats = false;
  const double eta = 0.2;
  std::vector<std::complex<double>> ws;
  for(double w = -6.0; w <= 6.0 + 1e-9; w += 1.0) ws.emplace_back(w, eta);

  const auto w3 = macis::make_orbital_cartan_weights(2, 1, 3);
  auto op = [&](const macis::wfn_t<N>& d) {
    return macis::weighted_imp_value<N>(d, w3, macis::DiagChannel::Charge,
                                        n_imp, n_active);
  };

  auto R_plain = macis::RunResolventDiagonal<N, int32_t>(
      psi0, ham_gen, dets, op, n_imp, E0, ws, settings,
      /*subtract_mean=*/false);
  auto R_delta = macis::RunResolventDiagonal<N, int32_t>(
      psi0, ham_gen, dets, op, n_imp, E0, ws, settings,
      /*subtract_mean=*/true);
  REQUIRE(R_plain.size() == ws.size());
  REQUIRE(R_delta.size() == ws.size());

  // v = O|psi0> and its mean <O> = <psi0|O|psi0> (psi0 is normalized).
  Eigen::VectorXd v(ndet);
  for(int k = 0; k < ndet; ++k) v(k) = psi0(k) * op(dets[k]);
  const double Omean = psi0.dot(v);
  const Eigen::VectorXd delta_v = v - Omean * psi0;
  const Eigen::VectorXd delta_overlaps =
      es.eigenvectors().transpose() * delta_v;

  // The fluctuation is orthogonal to the reference state by construction, so
  // the elastic (n = 0) overlap must vanish.
  REQUIRE(std::abs(delta_overlaps(0)) < 1e-10);

  SECTION("delta resolvent equals the inelastic-only Lehmann sum") {
    for(size_t iw = 0; iw < ws.size(); ++iw) {
      std::complex<double> ref(0.0, 0.0);
      for(int n = 0; n < ndet; ++n) {
        const double wn = es.eigenvalues()(n) - E0;
        ref += (delta_overlaps(n) * delta_overlaps(n)) / (ws[iw] - wn);
      }
      REQUIRE(std::real(R_delta[iw]) ==
              Approx(std::real(ref)).epsilon(1e-6).margin(1e-8));
      REQUIRE(std::imag(R_delta[iw]) ==
              Approx(std::imag(ref)).epsilon(1e-6).margin(1e-8));
    }
  }

  SECTION("plain minus delta equals the elastic pole <O>^2 / w") {
    // The n = 0 term of the plain resolvent sits at w_0 = 0 with weight
    // |<0|O|psi0>|^2 = <O>^2; subtracting the mean removes exactly it.
    for(size_t iw = 0; iw < ws.size(); ++iw) {
      const std::complex<double> elastic = Omean * Omean / ws[iw];
      REQUIRE(std::real(R_plain[iw] - R_delta[iw]) ==
              Approx(std::real(elastic)).epsilon(1e-6).margin(1e-8));
      REQUIRE(std::imag(R_plain[iw] - R_delta[iw]) ==
              Approx(std::imag(elastic)).epsilon(1e-6).margin(1e-8));
    }
  }
}

TEST_CASE("Dynamical properties - orbital matrix resolvent vs Lehmann") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t n_imp = 2;
  const size_t n_active = 4;
  std::vector<double> T(n_active * n_active, 0.0);
  std::vector<double> V(n_active * n_active * n_active * n_active, 0.0);
  for(size_t p = 0; p < n_active; ++p) {
    T[p * n_active + p] = -1.0 - 0.1 * double(p);
    for(size_t q = 0; q < n_active; ++q)
      if(p != q) T[p * n_active + q] = -0.25;
    V[((p * n_active + p) * n_active + p) * n_active + p] = 1.0;
  }
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), n_active, n_active),
      macis::rank4_span<double>(V.data(), n_active, n_active, n_active,
                                n_active));
  std::vector<macis::wfn_t<N>> dets;
  for(int a0 = 0; a0 < 4; ++a0)
    for(int a1 = a0 + 1; a1 < 4; ++a1)
      for(int b0 = 0; b0 < 4; ++b0)
        for(int b1 = b0 + 1; b1 < 4; ++b1)
          dets.push_back(make_det({a0, a1}, {b0, b1}));
  const Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);
  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;

  const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, E0, ws, settings);
  REQUIRE(result.rank > 0);
  REQUIRE(result.rank <= n_imp * n_imp);
  REQUIRE(result.gram.isApprox(result.gram.transpose(), 1e-12));
  for(Eigen::Index pair = 0; pair < result.capture.size(); ++pair)
    REQUIRE(result.capture(pair) == Approx(1.0).margin(1e-12));

  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);
  Eigen::MatrixXd seeds(dets.size(), n_imp * n_imp);
  for(size_t mu = 0; mu < n_imp; ++mu)
    for(size_t nu = 0; nu < n_imp; ++nu)
      seeds.col(mu * n_imp + nu) =
          macis::apply_spin_bilinear<N>(psi0, dets, index, mu, nu);
  const Eigen::MatrixXd overlaps = es.eigenvectors().transpose() * seeds;
  for(size_t iw = 0; iw < ws.size(); ++iw)
    for(size_t k = 0; k < n_imp * n_imp; ++k)
      for(size_t l = 0; l < n_imp * n_imp; ++l) {
        std::complex<double> reference(0.0, 0.0);
        for(Eigen::Index state = 0; state < es.eigenvalues().size(); ++state)
          reference += overlaps(state, k) * overlaps(state, l) /
                       (ws[iw] - (es.eigenvalues()(state) - E0));
        const auto value = result.resolvent[iw][k * n_imp * n_imp + l];
        REQUIRE(std::real(value) ==
                Approx(std::real(reference)).epsilon(1e-6).margin(1e-8));
        REQUIRE(std::imag(value) ==
                Approx(std::imag(reference)).epsilon(1e-6).margin(1e-8));
      }
}

TEST_CASE("Dynamical properties - orbital spin bilinears and capture") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const auto det0 = make_det({1}, {0});
  const auto det1 = make_det({0}, {0});
  const std::vector<macis::wfn_t<N>> dets = {det0, det1};
  const Eigen::VectorXd coeffs = (Eigen::Vector2d() << 0.3, -0.7).finished();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);

  const auto spin01 = macis::apply_spin_bilinear<N>(coeffs, dets, index, 0, 1);
  REQUIRE(spin01(0) == Approx(0.0).margin(1e-12));
  REQUIRE(spin01(1) == Approx(0.3).epsilon(1e-12));

  const auto diagonal = macis::apply_spin_bilinear<N>(coeffs, dets, index, 0, 0);
  const auto weighted = macis::apply_diagonal_operator<N>(
      coeffs, dets, [](const macis::wfn_t<N>& det) {
        return macis::weighted_imp_value<N>(det, {1.0, 0.0},
                                            macis::DiagChannel::Spin, 2, 2) *
               2.0;
      });
  REQUIRE((diagonal - weighted).squaredNorm() == Approx(0.0).margin(1e-12));
  REQUIRE(macis::spin_bilinear_capture_fraction<N>(coeffs, dets, index, 0, 1) ==
          Approx(1.0).margin(1e-12));

  const std::vector<macis::wfn_t<N>> truncated_dets = {det0};
  const Eigen::VectorXd truncated_coeffs =
      (Eigen::VectorXd(1) << 0.3).finished();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> truncated_index;
  truncated_index.emplace(det0, 0);
  REQUIRE(macis::spin_bilinear_capture_fraction<N>(
              truncated_coeffs, truncated_dets, truncated_index, 0, 1) ==
          Approx(0.0).margin(1e-12));
}

TEST_CASE(
    "Dynamical properties - spin bilinear signs across an occupied orbital") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // S_{20} on D = |alpha{0,1}, beta{0,1}>. In both spin blocks c^dagger_2 c_0
  // hops over the occupied orbital 1, so the fermionic sign is -1:
  //   c^dagger_2 c_0 c^dagger_0 c^dagger_1 |vac> = -|1 2>.
  // The beta hop also passes the two alpha creators (even, no extra sign) and
  // picks up the -1 of the spin-down term, so the two images carry opposite
  // signs: S_{20}|D> = -|alpha{1,2}, beta{0,1}> + |alpha{0,1}, beta{1,2}>.
  const auto D = make_det({0, 1}, {0, 1});
  const auto Da = make_det({1, 2}, {0, 1});
  const auto Db = make_det({0, 1}, {1, 2});
  const std::vector<macis::wfn_t<N>> dets = {D, Da, Db};
  const Eigen::VectorXd coeffs =
      (Eigen::Vector3d() << 0.3, 0.5, -0.7).finished();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);

  // Da and Db also map under S_{20}, but only onto |alpha{1,2}, beta{1,2}>,
  // which is outside the basis and dropped.
  const auto s20 = macis::apply_spin_bilinear<N>(coeffs, dets, index, 2, 0);
  REQUIRE(s20(0) == Approx(0.0).margin(1e-12));
  REQUIRE(s20(1) == Approx(-0.3).epsilon(1e-12));
  REQUIRE(s20(2) == Approx(0.3).epsilon(1e-12));

  // The reverse hop S_{02} brings both images back onto D with the same
  // signs: -0.5 (alpha) + (-1)(-1)(-0.7) (beta) = -1.2.
  const auto s02 = macis::apply_spin_bilinear<N>(coeffs, dets, index, 0, 2);
  REQUIRE(s02(0) == Approx(-1.2).epsilon(1e-12));
  REQUIRE(s02(1) == Approx(0.0).margin(1e-12));
  REQUIRE(s02(2) == Approx(0.0).margin(1e-12));

  // Capture of S_{20}: in-basis images carry 0.3^2 + 0.3^2 = 0.18. The
  // leaked image X = |alpha{1,2}, beta{1,2}> is reached twice, from Da via the
  // beta hop (-1 * -1 * 0.5 = +0.5) and from Db via the alpha hop
  // (-1 * -0.7 = +0.7), which interfere to 1.2. Capture = 0.18 / (0.18 + 1.44)
  // = 1/9. Squaring before accumulating would give 0.18 / 0.92 instead.
  REQUIRE(macis::spin_bilinear_capture_fraction<N>(coeffs, dets, index, 2, 0) ==
          Approx(1.0 / 9.0).epsilon(1e-12));
}

TEST_CASE("Dynamical properties - spin bilinear adjoint on the FCI space") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // The FCI space is closed under every S_{mu nu}, so the matrices built by
  // applying S_{mu nu} to unit vectors must satisfy S_{mu nu}^T = S_{nu mu}.
  // This is independent of any wave function, unlike the Gram symmetry.
  const size_t n_orb = 4;
  const auto dets = half_filled_fci_dets();
  const Eigen::Index L = dets.size();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);

  auto op_matrix = [&](size_t mu, size_t nu) {
    Eigen::MatrixXd S(L, L);
    for(Eigen::Index j = 0; j < L; ++j)
      S.col(j) = macis::apply_spin_bilinear<N>(Eigen::VectorXd::Unit(L, j),
                                               dets, index, mu, nu);
    return S;
  };
  for(size_t mu = 0; mu < n_orb; ++mu)
    for(size_t nu = 0; nu < n_orb; ++nu) {
      const Eigen::MatrixXd S_munu = op_matrix(mu, nu);
      const Eigen::MatrixXd S_numu = op_matrix(nu, mu);
      REQUIRE((S_munu.transpose() - S_numu).cwiseAbs().maxCoeff() ==
              Approx(0.0).margin(1e-14));
      if(mu != nu) REQUIRE(S_munu.cwiseAbs().maxCoeff() > 0.5);
    }
}

TEST_CASE(
    "Dynamical properties - orbital matrix resolvent deflates the exact null "
    "direction") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // With n_imp = n_active every determinant has S_z^tot = 0, so
  // sum_mu S_{mu mu}|psi0> = 2 S_z^tot |psi0> = 0 exactly. The Gram matrix then
  // has one null eigenvalue, the pipeline must keep r = M - 1 modes, and the
  // back-transformed matrix must still match the dense Lehmann sum.
  const size_t n_imp = 4;
  const size_t M = n_imp * n_imp;
  std::vector<double> T, V;
  orbital_matrix_integrals(T, V);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), n_imp, n_imp),
      macis::rank4_span<double>(V.data(), n_imp, n_imp, n_imp, n_imp));
  auto dets = half_filled_fci_dets();
  const Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  const Eigen::MatrixXd seeds = spin_bilinear_seeds(psi0, dets, n_imp);
  Eigen::VectorXd trace = Eigen::VectorXd::Zero(dets.size());
  for(size_t mu = 0; mu < n_imp; ++mu) trace += seeds.col(mu * n_imp + mu);
  REQUIRE(trace.norm() == Approx(0.0).margin(1e-12));

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;
  const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, E0, ws, settings);
  REQUIRE(result.rank == M - 1);
  // SelfAdjointEigenSolver sorts ascending: the discarded mode is first.
  REQUIRE(result.gram_eigenvalues(0) ==
          Approx(0.0).margin(1e-12 * result.gram_eigenvalues.maxCoeff()));

  const Eigen::MatrixXd overlaps = es.eigenvectors().transpose() * seeds;
  for(size_t iw = 0; iw < ws.size(); ++iw)
    for(size_t k = 0; k < M; ++k)
      for(size_t l = 0; l < M; ++l) {
        const auto reference = lehmann_element(es, overlaps, k, l, E0, ws[iw]);
        const auto value = result.resolvent[iw][k * M + l];
        REQUIRE(std::real(value) ==
                Approx(std::real(reference)).epsilon(1e-6).margin(1e-8));
        REQUIRE(std::imag(value) ==
                Approx(std::imag(reference)).epsilon(1e-6).margin(1e-8));
      }
}

TEST_CASE(
    "Dynamical properties - orbital matrix diagonal block vs "
    "RunResolventWeighted") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // S_{mu mu} = 2 S_z^mu, so each diagonal-block element must equal 4x the
  // Spin-channel weighted resolvent, which is an independent code path
  // (decompose_det + continued fraction instead of bit flips + band Lanczos):
  //   R_{mu mu; mu mu}              = 4 R_w[e_mu]
  //   R_{00;00} + R_{11;11} + 2 R_{00;11} = 4 R_w[(1, 1)]
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t M = n_imp * n_imp;
  std::vector<double> T, V;
  orbital_matrix_integrals(T, V);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), n_active, n_active),
      macis::rank4_span<double>(V.data(), n_active, n_active, n_active,
                                n_active));
  auto dets = half_filled_fci_dets();
  const Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);
  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;

  const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, E0, ws, settings);
  auto weighted = [&](const std::vector<double>& w) {
    return macis::RunResolventWeighted<N, int32_t>(
        psi0, ham_gen, dets, w, macis::DiagChannel::Spin, n_imp, n_active, E0,
        ws, settings);
  };
  const auto R0 = weighted({1.0, 0.0});
  const auto R1 = weighted({0.0, 1.0});
  const auto R01 = weighted({1.0, 1.0});

  const size_t p00 = 0 * n_imp + 0;
  const size_t p11 = 1 * n_imp + 1;
  auto require_close = [](std::complex<double> value,
                          std::complex<double> reference) {
    REQUIRE(std::real(value) ==
            Approx(std::real(reference)).epsilon(1e-6).margin(1e-8));
    REQUIRE(std::imag(value) ==
            Approx(std::imag(reference)).epsilon(1e-6).margin(1e-8));
  };
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    const auto& R = result.resolvent[iw];
    require_close(R[p00 * M + p00], 4.0 * R0[iw]);
    require_close(R[p11 * M + p11], 4.0 * R1[iw]);
    require_close(R[p00 * M + p00] + R[p11 * M + p11] + 2.0 * R[p00 * M + p11],
                  4.0 * R01[iw]);
  }
}

TEST_CASE("Dynamical properties - orbital matrix resolvent sum rule") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Zeroth moment: z R(z) = G + M1 / z + M2 / z^2 + ..., with G the Gram
  // matrix. At z = iY the M1 term is purely imaginary, so Re[z R(z)] = G up to
  // O(|M2| / Y^2), i.e. every one of the M^2 elements is checked against a
  // quantity that needed no Lanczos at all.
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t M = n_imp * n_imp;
  std::vector<double> T, V;
  orbital_matrix_integrals(T, V);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), n_active, n_active),
      macis::rank4_span<double>(V.data(), n_active, n_active, n_active,
                                n_active));
  auto dets = half_filled_fci_dets();
  const Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);
  const std::complex<double> z(0.0, 1.0e6);
  macis::GFSettings settings;
  settings.nLanIts = 100;

  const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, E0, {z}, settings);
  const Eigen::MatrixXd seeds = spin_bilinear_seeds(psi0, dets, n_imp);
  const Eigen::MatrixXd gram = seeds.transpose() * seeds;
  REQUIRE((result.gram - gram).cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1e-12));
  for(size_t k = 0; k < M; ++k)
    for(size_t l = 0; l < M; ++l)
      REQUIRE(std::real(z * result.resolvent[0][k * M + l]) ==
              Approx(gram(k, l)).epsilon(1e-6).margin(1e-8));
}

TEST_CASE(
    "Dynamical properties - subtract_mean zeroes an operator proportional to "
    "the identity") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // O = c*I is the degenerate case: delta_O = 0 identically, so the resolvent
  // must be exactly zero everywhere rather than NaN from dividing by ||v|| = 0.
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t norb = n_active;

  std::vector<double> T(norb * norb, 0.0);
  std::vector<double> V(norb * norb * norb * norb, 0.0);
  for(size_t p = 0; p < norb; ++p) T[p * norb + p] = -1.0 - 0.1 * double(p);
  for(size_t p = 0; p < norb; ++p)
    V[((p * norb + p) * norb + p) * norb + p] = 1.0;

  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));

  std::vector<macis::wfn_t<N>> dets;
  std::vector<int> orbs = {0, 1, 2, 3};
  for(size_t a0 = 0; a0 < orbs.size(); ++a0)
    for(size_t a1 = a0 + 1; a1 < orbs.size(); ++a1)
      for(size_t b0 = 0; b0 < orbs.size(); ++b0)
        for(size_t b1 = b0 + 1; b1 < orbs.size(); ++b1)
          dets.push_back(make_det({orbs[a0], orbs[a1]}, {orbs[b0], orbs[b1]}));

  Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  macis::GFSettings settings;
  settings.nLanIts = 200;
  settings.saveGFmats = false;
  std::vector<std::complex<double>> ws;
  for(double w = -6.0; w <= 6.0 + 1e-9; w += 1.0) ws.emplace_back(w, 0.2);

  auto const_op = [](const macis::wfn_t<N>&) { return 3.0; };
  auto R = macis::RunResolventDiagonal<N, int32_t>(
      psi0, ham_gen, dets, const_op, n_imp, E0, ws, settings,
      /*subtract_mean=*/true);

  REQUIRE(R.size() == ws.size());
  for(const auto& r : R) REQUIRE(std::abs(r) == Approx(0.0).margin(1e-14));
}

TEST_CASE(
    "Dynamical properties - orbital matrix resolvent subtract_mean cancels "
    "the elastic pole") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // A (2 alpha, 1 beta) doublet has <S_{mu mu}> != 0, so the plain matrix
  // resolvent carries the elastic n = 0 Lehmann term m_k m_l / z with
  // m_k = <psi0|phi_k>. The fluctuation seeds are orthogonal to psi0 and
  // unchanged along every other eigenvector, so the difference is exactly that
  // term, and the fluctuation Gram matrix is G - m m^T.
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t M = n_imp * n_imp;
  std::vector<double> T, V;
  orbital_matrix_integrals(T, V);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), n_active, n_active),
      macis::rank4_span<double>(V.data(), n_active, n_active, n_active,
                                n_active));
  std::vector<macis::wfn_t<N>> dets;
  for(int a0 = 0; a0 < 4; ++a0)
    for(int a1 = a0 + 1; a1 < 4; ++a1)
      for(int b0 = 0; b0 < 4; ++b0) dets.push_back(make_det({a0, a1}, {b0}));
  const Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);
  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;

  const Eigen::MatrixXd seeds = spin_bilinear_seeds(psi0, dets, n_imp);
  const Eigen::VectorXd m = seeds.transpose() * psi0;
  REQUIRE(m.cwiseAbs().maxCoeff() > 1e-3);

  const auto plain = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, E0, ws, settings);
  const auto delta = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, E0, ws, settings, /*subtract_mean=*/true);

  const Eigen::MatrixXd gram_delta = seeds.transpose() * seeds - m * m.transpose();
  REQUIRE((delta.gram - gram_delta).cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1e-12));

  for(size_t iw = 0; iw < ws.size(); ++iw)
    for(size_t k = 0; k < M; ++k)
      for(size_t l = 0; l < M; ++l) {
        const std::complex<double> elastic = m(k) * m(l) / ws[iw];
        const auto diff = plain.resolvent[iw][k * M + l] -
                          delta.resolvent[iw][k * M + l];
        REQUIRE(std::real(diff) ==
                Approx(std::real(elastic)).epsilon(1e-6).margin(1e-8));
        REQUIRE(std::imag(diff) ==
                Approx(std::imag(elastic)).epsilon(1e-6).margin(1e-8));
      }
}
