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
#include <macis/csr_hamiltonian.hpp>
#include <macis/gf/dynamical_properties.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <map>
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

// All n_imp^2 seeds O_{mu nu}|psi0> as columns, pair index mu * n_imp + nu,
// with O = S (Spin) or N (Charge).
Eigen::MatrixXd bilinear_seeds(
    const Eigen::VectorXd& psi0, const std::vector<macis::wfn_t<N>>& dets,
    size_t n_imp, macis::DiagChannel ch = macis::DiagChannel::Spin) {
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);
  Eigen::MatrixXd seeds(dets.size(), n_imp * n_imp);
  for(size_t mu = 0; mu < n_imp; ++mu)
    for(size_t nu = 0; nu < n_imp; ++nu)
      seeds.col(mu * n_imp + nu) =
          macis::apply_orbital_bilinear<N>(psi0, dets, index, mu, nu, ch);
  return seeds;
}

// Matrix of O_{mu nu} over `dets`, built column by column from unit vectors.
Eigen::MatrixXd bilinear_matrix(const std::vector<macis::wfn_t<N>>& dets,
                                size_t mu, size_t nu, macis::DiagChannel ch) {
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);
  const Eigen::Index L = dets.size();
  Eigen::MatrixXd O(L, L);
  for(Eigen::Index j = 0; j < L; ++j)
    O.col(j) = macis::apply_orbital_bilinear<N>(Eigen::VectorXd::Unit(L, j),
                                                dets, index, mu, nu, ch);
  return O;
}

// Every element of two flattened M x M resolvents agrees.
void require_resolvents_close(const std::vector<std::complex<double>>& a,
                              const std::vector<std::complex<double>>& b) {
  REQUIRE(a.size() == b.size());
  for(size_t i = 0; i < a.size(); ++i) {
    REQUIRE(std::real(a[i]) ==
            Approx(std::real(b[i])).epsilon(1e-6).margin(1e-8));
    REQUIRE(std::imag(a[i]) ==
            Approx(std::imag(b[i])).epsilon(1e-6).margin(1e-8));
  }
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
      psi0, ham_gen, dets, w3, macis::DiagChannel::Charge, n_imp, n_active, E0,
      ws, settings);
  REQUIRE(R.size() == ws.size());

  // Exact Lehmann reference: v = T^3 |psi0>.
  Eigen::VectorXd v(ndet);
  for(int k = 0; k < ndet; ++k)
    v(k) =
        psi0(k) * macis::weighted_imp_value<N>(
                      dets[k], w3, macis::DiagChannel::Charge, n_imp, n_active);

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

TEST_CASE("Dynamical properties - subtract_mean cancels the elastic pole") {
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
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, settings);
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
      seeds.col(mu * n_imp + nu) = macis::apply_orbital_bilinear<N>(
          psi0, dets, index, mu, nu, macis::DiagChannel::Spin);
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

  const auto spin01 = macis::apply_orbital_bilinear<N>(
      coeffs, dets, index, 0, 1, macis::DiagChannel::Spin);
  REQUIRE(spin01(0) == Approx(0.0).margin(1e-12));
  REQUIRE(spin01(1) == Approx(0.3).epsilon(1e-12));

  const auto diagonal = macis::apply_orbital_bilinear<N>(
      coeffs, dets, index, 0, 0, macis::DiagChannel::Spin);
  const auto weighted = macis::apply_diagonal_operator<N>(
      coeffs, dets, [](const macis::wfn_t<N>& det) {
        return macis::weighted_imp_value<N>(det, {1.0, 0.0},
                                            macis::DiagChannel::Spin, 2, 2) *
               2.0;
      });
  REQUIRE((diagonal - weighted).squaredNorm() == Approx(0.0).margin(1e-12));
  REQUIRE(macis::orbital_bilinear_captured_fraction<N>(
              coeffs, dets, index, 0, 1, macis::DiagChannel::Spin) ==
          Approx(1.0).margin(1e-12));

  const std::vector<macis::wfn_t<N>> truncated_dets = {det0};
  const Eigen::VectorXd truncated_coeffs =
      (Eigen::VectorXd(1) << 0.3).finished();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>>
      truncated_index;
  truncated_index.emplace(det0, 0);
  REQUIRE(macis::orbital_bilinear_captured_fraction<N>(
              truncated_coeffs, truncated_dets, truncated_index, 0, 1,
              macis::DiagChannel::Spin) == Approx(0.0).margin(1e-12));
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
  const auto s20 = macis::apply_orbital_bilinear<N>(coeffs, dets, index, 2, 0,
                                                    macis::DiagChannel::Spin);
  REQUIRE(s20(0) == Approx(0.0).margin(1e-12));
  REQUIRE(s20(1) == Approx(-0.3).epsilon(1e-12));
  REQUIRE(s20(2) == Approx(0.3).epsilon(1e-12));

  // The reverse hop S_{02} brings both images back onto D with the same
  // signs: -0.5 (alpha) + (-1)(-1)(-0.7) (beta) = -1.2.
  const auto s02 = macis::apply_orbital_bilinear<N>(coeffs, dets, index, 0, 2,
                                                    macis::DiagChannel::Spin);
  REQUIRE(s02(0) == Approx(-1.2).epsilon(1e-12));
  REQUIRE(s02(1) == Approx(0.0).margin(1e-12));
  REQUIRE(s02(2) == Approx(0.0).margin(1e-12));

  // Capture of S_{20}: in-basis images carry 0.3^2 + 0.3^2 = 0.18. The
  // leaked image X = |alpha{1,2}, beta{1,2}> is reached twice, from Da via the
  // beta hop (-1 * -1 * 0.5 = +0.5) and from Db via the alpha hop
  // (-1 * -0.7 = +0.7), which interfere to 1.2. Capture = 0.18 / (0.18 + 1.44)
  // = 1/9. Squaring before accumulating would give 0.18 / 0.92 instead.
  REQUIRE(macis::orbital_bilinear_captured_fraction<N>(
              coeffs, dets, index, 2, 0, macis::DiagChannel::Spin) ==
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
      S.col(j) = macis::apply_orbital_bilinear<N>(Eigen::VectorXd::Unit(L, j),
                                                  dets, index, mu, nu,
                                                  macis::DiagChannel::Spin);
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

  const Eigen::MatrixXd seeds = bilinear_seeds(psi0, dets, n_imp);
  Eigen::VectorXd trace = Eigen::VectorXd::Zero(dets.size());
  for(size_t mu = 0; mu < n_imp; ++mu) trace += seeds.col(mu * n_imp + mu);
  REQUIRE(trace.norm() == Approx(0.0).margin(1e-12));

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;
  const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, settings);
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
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, settings);
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
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, {z}, settings);
  const Eigen::MatrixXd seeds = bilinear_seeds(psi0, dets, n_imp);
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

  const Eigen::MatrixXd seeds = bilinear_seeds(psi0, dets, n_imp);
  const Eigen::VectorXd m = seeds.transpose() * psi0;
  REQUIRE(m.cwiseAbs().maxCoeff() > 1e-3);

  const auto plain = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, settings);
  const auto delta = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, settings,
      /*subtract_mean=*/true);

  const Eigen::MatrixXd gram_delta =
      seeds.transpose() * seeds - m * m.transpose();
  REQUIRE((delta.gram - gram_delta).cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1e-12));

  for(size_t iw = 0; iw < ws.size(); ++iw)
    for(size_t k = 0; k < M; ++k)
      for(size_t l = 0; l < M; ++l) {
        const std::complex<double> elastic = m(k) * m(l) / ws[iw];
        const auto diff =
            plain.resolvent[iw][k * M + l] - delta.resolvent[iw][k * M + l];
        REQUIRE(std::real(diff) ==
                Approx(std::real(elastic)).epsilon(1e-6).margin(1e-8));
        REQUIRE(std::imag(diff) ==
                Approx(std::imag(elastic)).epsilon(1e-6).margin(1e-8));
      }
}

TEST_CASE(
    "Dynamical properties - charge bilinear signs across an occupied orbital") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Same determinants as the spin test above, but N_{20} has the same sign in
  // both spin blocks: N_{20}|D> = -|alpha{1,2}, beta{0,1}> - |alpha{0,1},
  // beta{1,2}>.
  const auto D = make_det({0, 1}, {0, 1});
  const auto Da = make_det({1, 2}, {0, 1});
  const auto Db = make_det({0, 1}, {1, 2});
  const std::vector<macis::wfn_t<N>> dets = {D, Da, Db};
  const Eigen::VectorXd coeffs =
      (Eigen::Vector3d() << 0.3, 0.5, -0.7).finished();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);
  const auto charge = macis::DiagChannel::Charge;

  const auto n20 =
      macis::apply_orbital_bilinear<N>(coeffs, dets, index, 2, 0, charge);
  REQUIRE(n20(0) == Approx(0.0).margin(1e-12));
  REQUIRE(n20(1) == Approx(-0.3).epsilon(1e-12));
  REQUIRE(n20(2) == Approx(-0.3).epsilon(1e-12));

  // Reverse hop onto D: -0.5 (alpha) + (-1)(-0.7) (beta) = +0.2, where the
  // spin channel gives -1.2.
  const auto n02 =
      macis::apply_orbital_bilinear<N>(coeffs, dets, index, 0, 2, charge);
  REQUIRE(n02(0) == Approx(0.2).epsilon(1e-12));
  REQUIRE(n02(1) == Approx(0.0).margin(1e-12));
  REQUIRE(n02(2) == Approx(0.0).margin(1e-12));

  // Capture of N_{20}: the leaked image X = |alpha{1,2}, beta{1,2}> gets
  // -0.5 from Da (beta hop, no spin sign) and +0.7 from Db, which now
  // interfere destructively to 0.2. Capture = 0.18 / (0.18 + 0.04) = 9/11,
  // against 1/9 in the spin channel: the channel changes the interference.
  REQUIRE(macis::orbital_bilinear_captured_fraction<N>(coeffs, dets, index, 2,
                                                       0, charge) ==
          Approx(9.0 / 11.0).epsilon(1e-12));

  // N_{mu mu} = n_mu matches the Charge channel of weighted_imp_value with no
  // extra factor (the spin channel needs 2x).
  const auto n00 =
      macis::apply_orbital_bilinear<N>(coeffs, dets, index, 0, 0, charge);
  const auto weighted = macis::apply_diagonal_operator<N>(
      coeffs, dets, [](const macis::wfn_t<N>& det) {
        return macis::weighted_imp_value<N>(det, {1.0, 0.0, 0.0},
                                            macis::DiagChannel::Charge, 3, 3);
      });
  REQUIRE((n00 - weighted).squaredNorm() == Approx(0.0).margin(1e-12));
}

TEST_CASE(
    "Dynamical properties - charge bilinear adjoint and trace on the FCI "
    "space") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // N_{mu nu}^T = N_{nu mu} on the closed FCI space, and with every orbital
  // included sum_mu N_{mu mu} = N_tot = 4 on the (2 alpha, 2 beta) sector.
  const size_t n_orb = 4;
  const auto dets = half_filled_fci_dets();
  const Eigen::Index L = dets.size();
  const auto charge = macis::DiagChannel::Charge;
  Eigen::MatrixXd trace = Eigen::MatrixXd::Zero(L, L);
  for(size_t mu = 0; mu < n_orb; ++mu) {
    trace += bilinear_matrix(dets, mu, mu, charge);
    for(size_t nu = 0; nu < n_orb; ++nu) {
      const Eigen::MatrixXd N_munu = bilinear_matrix(dets, mu, nu, charge);
      const Eigen::MatrixXd N_numu = bilinear_matrix(dets, nu, mu, charge);
      REQUIRE((N_munu.transpose() - N_numu).cwiseAbs().maxCoeff() ==
              Approx(0.0).margin(1e-14));
      if(mu != nu) REQUIRE(N_munu.cwiseAbs().maxCoeff() > 0.5);
    }
  }
  REQUIRE(
      (trace - 4.0 * Eigen::MatrixXd::Identity(L, L)).cwiseAbs().maxCoeff() ==
      Approx(0.0).margin(1e-14));
}

TEST_CASE(
    "Dynamical properties - charge matrix resolvent trace direction is "
    "parallel to psi0") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // With n_imp = n_active, sum_mu N_{mu mu}|psi0> = N_tot|psi0> = 4|psi0>:
  // not a null direction (unlike the spin trace), so all M modes are kept,
  // but that direction is pure elastic weight. With subtract_mean it becomes
  // exactly zero and is deflated, r = M - 1. Both must match Lehmann.
  const size_t n_imp = 4;
  const size_t M = n_imp * n_imp;
  const auto charge = macis::DiagChannel::Charge;
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

  const Eigen::MatrixXd seeds = bilinear_seeds(psi0, dets, n_imp, charge);
  Eigen::VectorXd trace = Eigen::VectorXd::Zero(dets.size());
  for(size_t mu = 0; mu < n_imp; ++mu) trace += seeds.col(mu * n_imp + mu);
  REQUIRE((trace - 4.0 * psi0).norm() == Approx(0.0).margin(1e-12));

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;
  const auto plain = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, charge, E0, ws, settings);
  const auto delta = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, charge, E0, ws, settings,
      /*subtract_mean=*/true);
  REQUIRE(plain.rank == M);
  REQUIRE(delta.rank == M - 1);
  REQUIRE(delta.gram_eigenvalues(0) ==
          Approx(0.0).margin(1e-12 * delta.gram_eigenvalues.maxCoeff()));

  const Eigen::VectorXd m = seeds.transpose() * psi0;
  const Eigen::MatrixXd delta_seeds = seeds - psi0 * m.transpose();
  const Eigen::MatrixXd overlaps = es.eigenvectors().transpose() * seeds;
  const Eigen::MatrixXd delta_overlaps =
      es.eigenvectors().transpose() * delta_seeds;
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    std::vector<std::complex<double>> ref(M * M), delta_ref(M * M);
    for(size_t k = 0; k < M; ++k)
      for(size_t l = 0; l < M; ++l) {
        ref[k * M + l] = lehmann_element(es, overlaps, k, l, E0, ws[iw]);
        delta_ref[k * M + l] =
            lehmann_element(es, delta_overlaps, k, l, E0, ws[iw]);
      }
    require_resolvents_close(plain.resolvent[iw], ref);
    require_resolvents_close(delta.resolvent[iw], delta_ref);
  }
}

TEST_CASE(
    "Dynamical properties - charge matrix diagonal block vs "
    "RunResolventWeighted") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // N_{mu mu} = n_mu, so the diagonal block equals the Charge-channel weighted
  // resolvent with factor 1 (not the 4 of the spin channel):
  //   R_{mu mu; mu mu}                      = R_w[e_mu]
  //   R_{00;00} + R_{11;11} +/- 2 R_{00;11} = R_w[(1, +/-1)]
  // (1, -1) is the orbital T^3 of GF.TZ_RESOLVENT.
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

  for(bool subtract_mean : {false, true}) {
    const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
        psi0, ham_gen, dets, n_imp, macis::DiagChannel::Charge, E0, ws,
        settings, subtract_mean);
    auto weighted = [&](const std::vector<double>& w) {
      return macis::RunResolventWeighted<N, int32_t>(
          psi0, ham_gen, dets, w, macis::DiagChannel::Charge, n_imp, n_active,
          E0, ws, settings, subtract_mean);
    };
    const auto R0 = weighted({1.0, 0.0});
    const auto R1 = weighted({0.0, 1.0});
    const auto Rp = weighted({1.0, 1.0});
    const auto Rm = weighted({1.0, -1.0});

    const size_t p00 = 0 * n_imp + 0;
    const size_t p11 = 1 * n_imp + 1;
    for(size_t iw = 0; iw < ws.size(); ++iw) {
      const auto& R = result.resolvent[iw];
      const auto cross = R[p00 * M + p11];
      require_resolvents_close(
          {R[p00 * M + p00], R[p11 * M + p11],
           R[p00 * M + p00] + R[p11 * M + p11] + 2.0 * cross,
           R[p00 * M + p00] + R[p11 * M + p11] - 2.0 * cross},
          {R0[iw], R1[iw], Rp[iw], Rm[iw]});
    }
  }
}

TEST_CASE(
    "Dynamical properties - charge matrix resolvent elastic pole and sum "
    "rule") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // <N_{mu mu}> is an orbital occupation (order 1), so unlike the spin
  // singlet the plain charge resolvent always carries the elastic term
  // m_k m_l / z. Check it is removed exactly by subtract_mean, that both
  // Gram matrices are the zeroth moment Re[z R(z)] at large |z|, and that the
  // plain result matches the dense Lehmann sum.
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t M = n_imp * n_imp;
  const auto charge = macis::DiagChannel::Charge;
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
  macis::GFSettings settings;
  settings.nLanIts = 100;

  const Eigen::MatrixXd seeds = bilinear_seeds(psi0, dets, n_imp, charge);
  const Eigen::VectorXd m = seeds.transpose() * psi0;
  REQUIRE(m(0) > 0.5);  // <n_0>
  REQUIRE(m(3) > 0.5);  // <n_1>
  const Eigen::MatrixXd gram = seeds.transpose() * seeds;
  const Eigen::MatrixXd gram_delta = gram - m * m.transpose();

  const std::complex<double> z_large(0.0, 1.0e6);
  const std::vector<std::complex<double>> ws = {
      {0.5, 0.2}, {2.0, 0.2}, z_large};
  const auto plain = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, charge, E0, ws, settings);
  const auto delta = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, charge, E0, ws, settings,
      /*subtract_mean=*/true);
  REQUIRE((plain.gram - gram).cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1e-12));
  REQUIRE((delta.gram - gram_delta).cwiseAbs().maxCoeff() ==
          Approx(0.0).margin(1e-12));
  for(Eigen::Index pair = 0; pair < plain.capture.size(); ++pair)
    REQUIRE(plain.capture(pair) == Approx(1.0).margin(1e-12));

  const Eigen::MatrixXd overlaps = es.eigenvectors().transpose() * seeds;
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    std::vector<std::complex<double>> ref(M * M), diff(M * M), elastic(M * M);
    for(size_t k = 0; k < M; ++k)
      for(size_t l = 0; l < M; ++l) {
        ref[k * M + l] = lehmann_element(es, overlaps, k, l, E0, ws[iw]);
        diff[k * M + l] =
            plain.resolvent[iw][k * M + l] - delta.resolvent[iw][k * M + l];
        elastic[k * M + l] = m(k) * m(l) / ws[iw];
      }
    require_resolvents_close(plain.resolvent[iw], ref);
    require_resolvents_close(diff, elastic);
  }

  const size_t iz = ws.size() - 1;
  for(size_t k = 0; k < M; ++k)
    for(size_t l = 0; l < M; ++l) {
      REQUIRE(std::real(z_large * plain.resolvent[iz][k * M + l]) ==
              Approx(gram(k, l)).epsilon(1e-6).margin(1e-8));
      REQUIRE(std::real(z_large * delta.resolvent[iz][k * M + l]) ==
              Approx(gram_delta(k, l)).epsilon(1e-6).margin(1e-8));
    }
}

TEST_CASE(
    "Dynamical properties - spin and charge matrix resolvents agree at the "
    "SU(4) point") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Two degenerate impurity orbitals (0, 1), each hybridised with its own
  // bath orbital (2, 3), with density-density interaction U' = U, J = 0 on the
  // impurity: H_int = U/2 N_imp (N_imp - 1) is SU(4) invariant. For an SU(4)
  // singlet ground state every traceless one-body bilinear shares one
  // resolvent function, normalised by tr(T^dagger T). S_{01} and N_{01} both
  // have norm 2, so
  //   R^S_{01;01} = R^N_{01;01},
  // and S_00 - S_11, N_00 - N_11, S_00 + S_11 (norm 4) each give 2 R_{01;01}.
  // This ties the spin-channel and charge-channel signs and normalizations
  // together, which no single-channel test can do.
  const size_t n_imp = 2;
  const size_t n = 4;
  const size_t M = n_imp * n_imp;
  const double U = 1.0, eps_imp = -0.5, eps_bath = 0.3, hyb = 0.4;
  std::vector<double> T(n * n, 0.0), V(n * n * n * n, 0.0);
  for(size_t mu = 0; mu < n_imp; ++mu) {
    const size_t b = mu + n_imp;
    T[mu * n + mu] = eps_imp;
    T[b * n + b] = eps_bath;
    T[mu * n + b] = T[b * n + mu] = hyb;
  }
  // Chemist's notation (pq|rs): (pp|pp) = U, (pp|qq) = U' = U, no exchange.
  for(size_t p = 0; p < n_imp; ++p)
    for(size_t q = 0; q < n_imp; ++q) V[((p * n + p) * n + q) * n + q] = U;
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(macis::matrix_span<double>(T.data(), n, n),
                         macis::rank4_span<double>(V.data(), n, n, n, n));
  auto dets = half_filled_fci_dets();
  const Eigen::MatrixXd Hd = dense_hamiltonian(dets, ham_gen);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  // The identity needs a single (non-degenerate) SU(4) singlet ground state.
  REQUIRE(es.eigenvalues()(1) - es.eigenvalues()(0) > 1e-3);
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;
  const auto spin = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, settings);
  const auto charge = macis::RunResolventOrbitalMatrix<N, int32_t>(
      psi0, ham_gen, dets, n_imp, macis::DiagChannel::Charge, E0, ws, settings);

  const size_t p00 = 0, p01 = 1, p10 = 2, p11 = 3;
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    const auto& S = spin.resolvent[iw];
    const auto& Nr = charge.resolvent[iw];
    const auto f = S[p01 * M + p01];
    REQUIRE(std::abs(f) > 1e-3);  // the check must not be 0 == 0
    auto diag = [&](const std::vector<std::complex<double>>& R, double sign) {
      return R[p00 * M + p00] + R[p11 * M + p11] +
             2.0 * sign * R[p00 * M + p11];
    };
    require_resolvents_close(
        {Nr[p01 * M + p01], S[p10 * M + p10], Nr[p10 * M + p10], diag(S, -1.0),
         diag(Nr, -1.0), diag(S, +1.0)},
        {f, f, f, 2.0 * f, 2.0 * f, 2.0 * f});
  }
}

namespace {

// Two impurity orbitals (0, 1), each hybridised only with its own bath orbital
// (2, 3), with density-density U (intra) and U' (inter), J = 0. Every
// one-body term stays inside one flavor {mu, mu + 2}, so H conserves the
// electron count of each flavor and spin.
void flavor_diagonal_integrals(std::vector<double>& T, std::vector<double>& V,
                               bool degenerate) {
  const size_t n = 4, n_imp = 2;
  const double U = 1.0;
  const double Up = degenerate ? U : 0.6;
  const double eps_imp[2] = {-0.5, degenerate ? -0.5 : -0.3};
  const double eps_bath[2] = {0.3, degenerate ? 0.3 : 0.2};
  const double hyb[2] = {0.4, degenerate ? 0.4 : 0.55};
  T.assign(n * n, 0.0);
  V.assign(n * n * n * n, 0.0);
  for(size_t mu = 0; mu < n_imp; ++mu) {
    const size_t b = mu + n_imp;
    T[mu * n + mu] = eps_imp[mu];
    T[b * n + b] = eps_bath[mu];
    T[mu * n + b] = T[b * n + mu] = hyb[mu];
  }
  // Chemist's notation (pq|rs): (pp|pp) = U, (pp|qq) = U', no exchange.
  for(size_t p = 0; p < n_imp; ++p)
    for(size_t q = 0; q < n_imp; ++q)
      V[((p * n + p) * n + q) * n + q] = p == q ? U : Up;
}

// Flavor sector of a determinant: electron count of each flavor {f, f + 2}
// per spin, packed into one integer.
int flavor_sector(const macis::wfn_t<N>& det) {
  int key = 0;
  for(int spin = 0; spin < 2; ++spin)
    for(int f = 0; f < 2; ++f) {
      const int off = spin * int(N / 2);
      key = 4 * key + int(det.test(f + off)) + int(det.test(f + 2 + off));
    }
  return key;
}

// The determinants of the ground state's flavor sector, i.e. what an ASCI
// solve grown through H-connected determinants can reach, and psi0 on them.
std::vector<macis::wfn_t<N>> ground_sector_dets(
    const std::vector<macis::wfn_t<N>>& dets, const Eigen::VectorXd& psi0,
    Eigen::VectorXd& psi_sector) {
  Eigen::Index kmax;
  psi0.cwiseAbs().maxCoeff(&kmax);
  const int key = flavor_sector(dets[kmax]);
  std::vector<macis::wfn_t<N>> sector;
  std::vector<double> coeffs;
  for(size_t k = 0; k < dets.size(); ++k)
    if(flavor_sector(dets[k]) == key) {
      sector.push_back(dets[k]);
      coeffs.push_back(psi0(k));
    }
  psi_sector = Eigen::Map<Eigen::VectorXd>(coeffs.data(), coeffs.size());
  return sector;
}

// Full-FCI flavor-diagonal model: dense eigensystem plus the ground state
// restricted to its sector.
struct FlavorModel {
  std::vector<double> T, V;
  std::vector<macis::wfn_t<N>> fci, sector;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  double E0 = 0.0;
  Eigen::VectorXd psi0, psi_sector;
};

template <class Gen>
void solve_flavor_model(FlavorModel& m, Gen& ham_gen) {
  m.fci = half_filled_fci_dets();
  m.es.compute(dense_hamiltonian(m.fci, ham_gen));
  // A single ground state: at J = 0 a degeneracy across flavor sectors would
  // make psi0 a mixture of sectors.
  REQUIRE(m.es.eigenvalues()(1) - m.es.eigenvalues()(0) > 1e-3);
  m.E0 = m.es.eigenvalues()(0);
  m.psi0 = m.es.eigenvectors().col(0);
  m.sector = ground_sector_dets(m.fci, m.psi0, m.psi_sector);
  REQUIRE(m.psi_sector.norm() == Approx(1.0).epsilon(1e-12));
  REQUIRE(m.sector.size() < m.fci.size());
}

}  // namespace

TEST_CASE("Dynamical properties - orbital bilinear image reports leaks") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Same state as the sign test above: S_{20} leaks only
  // X = |alpha{1,2}, beta{1,2}>, with accumulated amplitude 0.5 + 0.7 = 1.2.
  const auto D = make_det({0, 1}, {0, 1});
  const auto Da = make_det({1, 2}, {0, 1});
  const auto Db = make_det({0, 1}, {1, 2});
  const std::vector<macis::wfn_t<N>> dets = {D, Da, Db};
  const Eigen::VectorXd coeffs =
      (Eigen::Vector3d() << 0.3, 0.5, -0.7).finished();
  std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> index;
  for(size_t k = 0; k < dets.size(); ++k) index.emplace(dets[k], k);

  const auto image = macis::orbital_bilinear_image<N>(coeffs, dets, index, 2, 0,
                                                      macis::DiagChannel::Spin);
  REQUIRE(image.capture == Approx(1.0 / 9.0).epsilon(1e-12));
  REQUIRE(image.leaked.size() == 1);
  REQUIRE(image.leaked[0] == make_det({1, 2}, {1, 2}));
  REQUIRE(image.leaked_amplitude[0] == Approx(1.2).epsilon(1e-12));

  // A diagonal seed never leaks.
  const auto diag = macis::orbital_bilinear_image<N>(coeffs, dets, index, 1, 1,
                                                     macis::DiagChannel::Spin);
  REQUIRE(diag.capture == 1.0);
  REQUIRE(diag.leaked.empty());
}

TEST_CASE("Dynamical properties - basis growth gating and merge") {
  ROOT_ONLY(MPI_COMM_WORLD);

  SECTION("leaked images of different seeds do not interfere") {
    // Two pairs leak into the same determinant with opposite amplitudes. A
    // sum would cancel them to 0, below any growth threshold.
    const auto X = make_det({1}, {2});
    macis::OrbitalBilinearImage<N> a, b;
    a.leaked = {X};
    a.leaked_amplitude = {0.3};
    b.leaked = {X};
    b.leaked_amplitude = {-0.3};
    std::map<macis::wfn_t<N>, double, macis::bitset_less_comparator<N>> merged;
    macis::merge_leaked_images(merged, a);
    macis::merge_leaked_images(merged, b);
    REQUIRE(merged.size() == 1);
    REQUIRE(merged.at(X) == Approx(0.3).epsilon(1e-15));
  }

  SECTION("only seeds above GFseedThres are grown, all seeds are kept") {
    const auto big = make_det({0}, {0});
    const auto small = make_det({3}, {3});
    const auto base = make_det({1}, {1});
    std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> exclude;
    exclude.emplace(base, 0);
    macis::GFSettings settings;
    settings.tot_SD = 1;
    settings.GFseedThres = 0.1;
    const std::vector<uint32_t> as_orbs = {0, 1, 2, 3};
    const auto found = macis::grow_basis_by_singles<N>(
        {big, small}, {0.5, 0.05}, exclude, as_orbs, 4, settings);
    std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>> pos;
    for(size_t k = 0; k < found.size(); ++k) pos.emplace(found[k], k);
    REQUIRE(pos.size() == found.size());  // no duplicates
    REQUIRE(found[0] == big);
    REQUIRE(found[1] == small);
    // big = |a{0}, b{0}> has 3 alpha and 3 beta singles, none of them the
    // base determinant (that one moves both spins).
    REQUIRE(found.size() == 2 + 6);
    REQUIRE(pos.count(make_det({2}, {3})) == 0);  // single of small only
    REQUIRE(pos.count(make_det({0}, {2})) == 1);
    REQUIRE(pos.count(base) == 0);

    // The excluded determinant is never added, even when reached.
    const auto from_base_neighbour = macis::grow_basis_by_singles<N>(
        {make_det({1}, {0})}, {1.0}, exclude, as_orbs, 4, settings);
    for(const auto& d : from_base_neighbour) REQUIRE(d != base);
  }

  SECTION("trunc_size caps the growth but never drops a seed") {
    macis::GFSettings settings;
    settings.tot_SD = 3;
    settings.GFseedThres = 0.0;
    settings.trunc_size = 1;
    const std::vector<uint32_t> as_orbs = {0, 1, 2, 3};
    const std::map<macis::wfn_t<N>, size_t, macis::bitset_less_comparator<N>>
        none;
    const auto seeds = std::vector<macis::wfn_t<N>>{
        make_det({0}, {0}), make_det({1}, {1}), make_det({2}, {2})};
    const auto found = macis::grow_basis_by_singles<N>(
        seeds, {1.0, 1.0, 1.0}, none, as_orbs, 4, settings);
    REQUIRE(found.size() == 3);
  }
}

TEST_CASE(
    "Dynamical properties - off-diagonal capture is exactly zero in a flavor "
    "sector") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // The diagnosis of PLAN_capture_basis_expansion.md: with a flavor-diagonal
  // bath at J = 0, a basis confined to the ground state's flavor sector
  // captures none of S_{mu nu}|psi0> / N_{mu nu}|psi0> for mu != nu, and all of
  // it for mu = nu. Without expansion the off-diagonal elements are zero.
  const size_t n_imp = 2, n = 4, M = n_imp * n_imp;
  FlavorModel m;
  flavor_diagonal_integrals(m.T, m.V, /*degenerate=*/false);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(macis::matrix_span<double>(m.T.data(), n, n),
                         macis::rank4_span<double>(m.V.data(), n, n, n, n));
  solve_flavor_model(m, ham_gen);

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 100;
  for(auto ch : {macis::DiagChannel::Spin, macis::DiagChannel::Charge}) {
    const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
        m.psi_sector, ham_gen, m.sector, n_imp, ch, m.E0, ws, settings);
    REQUIRE(result.base_size == m.sector.size());
    REQUIRE(result.expanded_size == m.sector.size());
    for(size_t mu = 0; mu < n_imp; ++mu)
      for(size_t nu = 0; nu < n_imp; ++nu) {
        const size_t pair = mu * n_imp + nu;
        REQUIRE(result.capture(pair) == (mu == nu ? 1.0 : 0.0));
        REQUIRE(result.capture_expanded(pair) == result.capture(pair));
        REQUIRE_FALSE(result.expanded[pair]);
        if(mu == nu) continue;
        for(size_t iw = 0; iw < ws.size(); ++iw)
          for(size_t l = 0; l < M; ++l)
            REQUIRE(std::abs(result.resolvent[iw][pair * M + l]) == 0.0);
      }
  }
}

TEST_CASE(
    "Dynamical properties - basis expansion recovers the exact orbital matrix "
    "resolvent") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Same flavor-sector basis, gate on: the leaked sectors are grown until
  // complete (tot_SD = 4 on 4 orbitals), so every element, off-diagonal ones
  // included, must match the dense Lehmann sum on the full FCI space.
  const size_t n_imp = 2, n = 4, M = n_imp * n_imp;
  FlavorModel m;
  flavor_diagonal_integrals(m.T, m.V, /*degenerate=*/false);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(macis::matrix_span<double>(m.T.data(), n, n),
                         macis::rank4_span<double>(m.V.data(), n, n, n, n));
  solve_flavor_model(m, ham_gen);

  const std::vector<std::complex<double>> ws = {
      {0.5, 0.2}, {2.0, 0.2}, {-1.0, 0.1}};
  macis::GFSettings settings;
  settings.nLanIts = 200;
  settings.orb_expand_basis = true;
  settings.tot_SD = 4;
  settings.GFseedThres = 0.0;
  settings.norbs = n;

  for(auto ch : {macis::DiagChannel::Spin, macis::DiagChannel::Charge})
    for(bool subtract_mean : {false, true}) {
      const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
          m.psi_sector, ham_gen, m.sector, n_imp, ch, m.E0, ws, settings,
          subtract_mean);
      REQUIRE(result.base_size == m.sector.size());
      REQUIRE(result.expanded_size > result.base_size);
      for(size_t mu = 0; mu < n_imp; ++mu)
        for(size_t nu = 0; nu < n_imp; ++nu) {
          const size_t pair = mu * n_imp + nu;
          REQUIRE(result.expanded[pair] == (mu != nu));
          REQUIRE(result.capture_expanded(pair) == 1.0);
        }

      const Eigen::MatrixXd seeds = bilinear_seeds(m.psi0, m.fci, n_imp, ch);
      Eigen::MatrixXd ref_seeds = seeds;
      if(subtract_mean) ref_seeds -= m.psi0 * (m.psi0.transpose() * seeds);
      const Eigen::MatrixXd overlaps =
          m.es.eigenvectors().transpose() * ref_seeds;
      REQUIRE((result.gram - ref_seeds.transpose() * ref_seeds)
                  .cwiseAbs()
                  .maxCoeff() == Approx(0.0).margin(1e-12));
      for(size_t iw = 0; iw < ws.size(); ++iw) {
        std::vector<std::complex<double>> ref(M * M);
        for(size_t k = 0; k < M; ++k)
          for(size_t l = 0; l < M; ++l)
            ref[k * M + l] =
                lehmann_element(m.es, overlaps, k, l, m.E0, ws[iw]);
        // The off-diagonal elements must be the nonzero exact ones.
        REQUIRE(std::abs(ref[1 * M + 1]) > 1e-3);
        require_resolvents_close(result.resolvent[iw], ref);
      }
    }
}

TEST_CASE(
    "Dynamical properties - expanded resolvent satisfies the SU(4) identity") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Degenerate flavors, U' = U, J = 0: SU(4). With the off-diagonal element
  // recovered by the expansion,
  //   R_{01;01} = (R_{00;00} + R_{11;11} - 2 R_{00;11}) / 2.
  const size_t n_imp = 2, n = 4, M = n_imp * n_imp;
  FlavorModel m;
  flavor_diagonal_integrals(m.T, m.V, /*degenerate=*/true);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(macis::matrix_span<double>(m.T.data(), n, n),
                         macis::rank4_span<double>(m.V.data(), n, n, n, n));
  solve_flavor_model(m, ham_gen);

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings settings;
  settings.nLanIts = 200;
  settings.orb_expand_basis = true;
  settings.tot_SD = 4;
  settings.GFseedThres = 0.0;
  settings.norbs = n;
  const auto result = macis::RunResolventOrbitalMatrix<N, int32_t>(
      m.psi_sector, ham_gen, m.sector, n_imp, macis::DiagChannel::Spin, m.E0,
      ws, settings);
  REQUIRE(result.capture(1) == 0.0);
  REQUIRE(result.capture_expanded(1) == 1.0);
  const size_t p00 = 0, p01 = 1, p10 = 2, p11 = 3;
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    const auto& R = result.resolvent[iw];
    REQUIRE(std::abs(R[p01 * M + p01]) > 1e-3);
    const auto rhs =
        0.5 * (R[p00 * M + p00] + R[p11 * M + p11] - 2.0 * R[p00 * M + p11]);
    require_resolvents_close({R[p01 * M + p01], R[p10 * M + p10]}, {rhs, rhs});
  }
}

TEST_CASE(
    "Dynamical properties - basis expansion off or not needed is "
    "bit-identical") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // On the full FCI space every capture is 1, so the gate marks nothing and
  // the result must be bit-for-bit the unexpanded one.
  const size_t n_imp = 2, n_active = 4;
  std::vector<double> T, V;
  orbital_matrix_integrals(T, V);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(
      macis::matrix_span<double>(T.data(), n_active, n_active),
      macis::rank4_span<double>(V.data(), n_active, n_active, n_active,
                                n_active));
  auto dets = half_filled_fci_dets();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
      dense_hamiltonian(dets, ham_gen));
  const double E0 = es.eigenvalues()(0);
  const Eigen::VectorXd psi0 = es.eigenvectors().col(0);
  const std::vector<std::complex<double>> ws = {{0.5, 0.2}, {2.0, 0.2}};
  macis::GFSettings off;
  off.nLanIts = 100;
  macis::GFSettings on = off;
  on.orb_expand_basis = true;
  on.orb_min_capture = 0.999;

  for(bool subtract_mean : {false, true}) {
    const auto a = macis::RunResolventOrbitalMatrix<N, int32_t>(
        psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, off,
        subtract_mean);
    const auto b = macis::RunResolventOrbitalMatrix<N, int32_t>(
        psi0, ham_gen, dets, n_imp, macis::DiagChannel::Spin, E0, ws, on,
        subtract_mean);
    REQUIRE(b.expanded_size == b.base_size);
    REQUIRE(a.rank == b.rank);
    REQUIRE(a.gram == b.gram);
    REQUIRE(a.capture == b.capture);
    REQUIRE(a.resolvent == b.resolvent);
  }
}

TEST_CASE("Dynamical properties - basis expansion norbs fallback") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // settings.norbs = 0 uses occs.size(): same basis and result as norbs = 4.
  // Without either the expansion cannot generate singles and refuses.
  const size_t n_imp = 2, n = 4;
  FlavorModel m;
  flavor_diagonal_integrals(m.T, m.V, /*degenerate=*/false);
  using generator_type = macis::DoubleLoopHamiltonianGenerator<N>;
  generator_type ham_gen(macis::matrix_span<double>(m.T.data(), n, n),
                         macis::rank4_span<double>(m.V.data(), n, n, n, n));
  solve_flavor_model(m, ham_gen);

  const std::vector<std::complex<double>> ws = {{0.5, 0.2}};
  macis::GFSettings explicit_norbs;
  explicit_norbs.nLanIts = 200;
  explicit_norbs.orb_expand_basis = true;
  explicit_norbs.tot_SD = 2;
  explicit_norbs.norbs = n;
  macis::GFSettings fallback = explicit_norbs;
  fallback.norbs = 0;
  const std::vector<double> occs(n, 0.5);  // every orbital active

  const auto a = macis::RunResolventOrbitalMatrix<N, int32_t>(
      m.psi_sector, ham_gen, m.sector, n_imp, macis::DiagChannel::Spin, m.E0,
      ws, explicit_norbs);
  const auto b = macis::RunResolventOrbitalMatrix<N, int32_t>(
      m.psi_sector, ham_gen, m.sector, n_imp, macis::DiagChannel::Spin, m.E0,
      ws, fallback, false, occs);
  REQUIRE(a.expanded_size > a.base_size);
  REQUIRE(a.expanded_size == b.expanded_size);
  REQUIRE(a.resolvent == b.resolvent);
  REQUIRE_THROWS(macis::RunResolventOrbitalMatrix<N, int32_t>(
      m.psi_sector, ham_gen, m.sector, n_imp, macis::DiagChannel::Spin, m.E0,
      ws, fallback));
}
