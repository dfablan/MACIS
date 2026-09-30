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
#include <macis/gf/gf.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>

#include "ut_common.hpp"

namespace {

constexpr size_t N = 64;

macis::wfn_t<N> make_det(const std::vector<int>& alpha_occ,
                         const std::vector<int>& beta_occ) {
  macis::wfn_t<N> det = 0;
  for(int o : alpha_occ) det.set(o);
  for(int o : beta_occ) det.set(o + N / 2);
  return det;
}

using GF_t = std::vector<std::vector<std::complex<double>>>;

void require_same_gf(const GF_t& a, const GF_t& b) {
  REQUIRE(a.size() == b.size());
  for(size_t iw = 0; iw < a.size(); ++iw) {
    REQUIRE(a[iw].size() == b[iw].size());
    for(size_t k = 0; k < a[iw].size(); ++k) {
      REQUIRE(std::real(a[iw][k]) ==
              Approx(std::real(b[iw][k])).epsilon(1e-12).margin(1e-12));
      REQUIRE(std::imag(a[iw][k]) ==
              Approx(std::imag(b[iw][k])).epsilon(1e-12).margin(1e-12));
    }
  }
}

}  // namespace

// The GF path builds its N+/-1 Hamiltonian with 64-bit CSR indices (the space
// can exceed INT32_MAX nonzeros). The result must not depend on the index
// width.
TEST_CASE("RunGFCalc - 32-bit and 64-bit sparse indices agree") {
  ROOT_ONLY(MPI_COMM_WORLD);

  const size_t norb = 4;
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

  // Full 2 alpha + 2 beta sector.
  std::vector<macis::wfn_t<N>> dets;
  for(int a0 = 0; a0 < 4; ++a0)
    for(int a1 = a0 + 1; a1 < 4; ++a1)
      for(int b0 = 0; b0 < 4; ++b0)
        for(int b1 = b0 + 1; b1 < 4; ++b1)
          dets.push_back(make_det({a0, a1}, {b0, b1}));
  const int ndet = int(dets.size());

  auto H = macis::make_csr_hamiltonian_block<int32_t>(
      dets.begin(), dets.end(), dets.begin(), dets.end(), ham_gen, 1e-16);
  Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(ndet, ndet);
  for(int i = 0; i < ndet; ++i)
    for(int p = H.rowptr()[i]; p < H.rowptr()[i + 1]; ++p)
      Hd(i, H.colind()[p]) = H.nzval()[p];
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
  const double E0 = es.eigenvalues()(0);
  Eigen::VectorXd psi0 = es.eigenvectors().col(0);

  macis::GFSettings settings;
  settings.norbs = norb;
  settings.trunc_size = 1000000;
  settings.tot_SD = 1;
  settings.GFseedThres = 1e-12;
  settings.asThres = 1e-4;
  settings.GF_orbs_basis = {0, 1, 2, 3};
  settings.is_up_basis = {true, true, true, true};
  settings.GF_orbs_comp = {0, 1};
  settings.is_up_comp = {true, true};
  settings.nLanIts = 100;
  settings.writeGF = false;
  const std::vector<double> occs(norb, 0.5);  // every orbital active
  const std::vector<std::complex<double>> ws = {
      {0.0, 0.157}, {0.0, 0.471}, {0.0, 1.5}, {0.5, 0.1}};

  for(bool band : {true, false}) {
    settings.use_bandLan = band;
    for(bool is_part : {true, false}) {
      GF_t gf32, gf64;
      std::vector<int> del32, del64;
      macis::RunGFCalc<N, int32_t>(gf32, psi0, ham_gen, dets, E0, is_part, ws,
                                   occs, settings, del32);
      macis::RunGFCalc<N, int64_t>(gf64, psi0, ham_gen, dets, E0, is_part, ws,
                                   occs, settings, del64);
      REQUIRE(del32 == del64);
      require_same_gf(gf32, gf64);

      // Guard against a trivially passing comparison of two zero GFs.
      double gmax = 0.;
      for(const auto& g : gf64)
        for(const auto& x : g) gmax = std::max(gmax, std::abs(x));
      REQUIRE(gmax > 1e-3);
    }
  }
}
