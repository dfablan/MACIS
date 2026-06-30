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
    for(int p = rowptr[i]; p < rowptr[i + 1]; ++p)
      Hd(i, colind[p]) = nzval[p];
  return Hd;
}

}  // namespace

TEST_CASE("Dynamical properties - sz_imp_value and apply_diagonal_operator") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t n_imp = 2;
  const size_t n_active = 4;

  // det A: impurity = (up:0, dn:1)  -> n_imp_up=1, n_imp_dn=1 -> Sz_imp = 0
  //        (bath orbital 2 up, 3 dn keeps total Sz = 0)
  auto detA = make_det(/*alpha*/ {0, 2}, /*beta*/ {1, 3});
  // det B: impurity = (up:0,1 ; dn: none) -> n_imp_up=2, n_imp_dn=0 -> Sz_imp=+1
  //        bath: (up: none ; dn: 2,3) so total Sz = 0
  auto detB = make_det(/*alpha*/ {0, 1}, /*beta*/ {2, 3});
  // det C: impurity = (up: none ; dn:0,1) -> n_imp_up=0, n_imp_dn=2 -> Sz_imp=-1
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

    REQUIRE(v(0) == Approx(0.3 * 0.0).margin(1e-12));   // Sz_imp(A)=0
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
  for(double w = -6.0; w <= 6.0 + 1e-9; w += 1.0)
    ws.emplace_back(w, eta);

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
    REQUIRE(std::real(R[iw]) == Approx(std::real(ref)).epsilon(1e-6).margin(1e-8));
    REQUIRE(std::imag(R[iw]) == Approx(std::imag(ref)).epsilon(1e-6).margin(1e-8));
  }

  SECTION("first spectral moment is non-negative (excitations above GS)") {
    // m1 = <v|(H - E0)|v> = sum_n |<n|v>|^2 (E_n - E0) >= 0.
    double m1 = 0.0;
    for(int n = 0; n < ndet; ++n)
      m1 += (overlaps(n) * overlaps(n)) * (es.eigenvalues()(n) - E0);
    REQUIRE(m1 >= -1e-10);
  }

  SECTION("retarded spectral function is non-negative: Im R(w) <= 0") {
    for(size_t iw = 0; iw < ws.size(); ++iw)
      REQUIRE(std::imag(R[iw]) <= 1e-8);
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
