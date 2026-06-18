#include <cmath>
#include <macis/observables/impurity_rdm.hpp>

#include "ut_common.hpp"

TEST_CASE("Impurity RDM - single determinant projector") {
  ROOT_ONLY(MPI_COMM_WORLD);

  constexpr size_t N = 128;
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t dim = 1 << (2 * n_imp);  // 16

  // alpha orbitals 0,1 occupied; beta empty; bath empty
  macis::wfn_t<N> det = 0;
  det.set(0);
  det.set(1);

  std::vector<macis::wfn_t<N>> dets = {det};
  std::vector<double> C = {1.0};

  // Identity rotation: impurity orbital i = active orbital i
  std::vector<double> orb_rot(n_imp * n_active, 0.0);
  orb_rot[0 * n_active + 0] = 1.0;
  orb_rot[1 * n_active + 1] = 1.0;

  auto rho = macis::compute_impurity_rdm_from_state<N>(n_imp, n_active, dets, C,
                                                       orb_rot);

  // Print the RDM for debugging
  std::cout << "Non zero elements of the impurity RDM:\n";
  for(size_t i = 0; i < dim; ++i) {
    for(size_t j = 0; j < dim; ++j) {
      if(std::abs(rho[i * dim + j]) > 1e-10) {
        std::cout << "rho[" << i << "][" << j << "] = " << rho[i * dim + j]
                  << "\n";
        std::cout << "Basis element i: " << std::bitset<4>(i)
                  << ", Basis element j: " << std::bitset<4>(j) << "\n";
      }
    }
  }

  SECTION("trace equals number of impurity electrons") {
    double tr = 0.0;
    for(size_t i = 0; i < dim; ++i) tr += rho[i * dim + i];
    REQUIRE(tr == Approx(2.0).epsilon(1e-10));
  }

  SECTION("idempotent: rho^2 == rho") {
    std::vector<double> rho2(dim * dim, 0.0);
    for(size_t i = 0; i < dim; ++i)
      for(size_t k = 0; k < dim; ++k)
        for(size_t j = 0; j < dim; ++j)
          rho2[i * dim + j] += rho[i * dim + k] * rho[k * dim + j];
    for(size_t idx = 0; idx < dim * dim; ++idx)
      REQUIRE(rho2[idx] == Approx(rho[idx]).epsilon(1e-10));
  }

  SECTION("rank-1: exactly one nonzero diagonal entry") {
    int nonzero = 0;
    for(size_t i = 0; i < dim; ++i)
      if(std::abs(rho[i * dim + i]) > 1e-10) ++nonzero;
    REQUIRE(nonzero == 1);
  }
}
