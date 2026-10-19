#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
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

  // Print the determinant for debugging
  std::cout << "Determinant bitset: " << det << "\n";
  // print the determinant in binary for debugging
  std::cout << "Determinant binary: ";
  for(int i = 0; i < 2 * n_active; ++i) {
    std::cout << det[i];
  }
  std::cout << "\n";

  std::vector<macis::wfn_t<N>> dets = {det};
  std::vector<double> C = {1.0};

  // Identity rotation: impurity orbital i = active orbital i
  std::vector<double> orb_rot(n_active * n_active, 0.0);
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
        std::cout << "Basis element i: " << std::bitset<dim>(i)
                  << ", Basis element j: " << std::bitset<dim>(j) << "\n";
      }
    }
  }

  SECTION("assert rho dimension") { REQUIRE(rho.size() == dim * dim); }

  SECTION("print rho diagonal") {
    std::cout << "Diagonal elements of the impurity RDM:\n";
    for(size_t i = 0; i < dim; ++i) {
      std::cout << "rho[" << i << "][" << i << "] = " << rho[i * dim + i]
                << "\n";
    }
  }

  SECTION("trace equals 1: Tr(rho) == 1") {
    double tr = 0.0;
    for(size_t i = 0; i < dim; ++i) tr += rho[i * dim + i];
    REQUIRE(tr == Approx(1.0).epsilon(1e-10));
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

TEST_CASE("Impurity RDM - block-sparse matches dense") {
  ROOT_ONLY(MPI_COMM_WORLD);

  constexpr size_t N = 64;
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t dim = 1 << (2 * n_imp);
  const uint64_t mask = (1ULL << n_imp) - 1;

  auto make_det = [&](int imp_up_orb, int bath_up_p, int imp_dn_orb,
                      int bath_dn_p) {
    macis::wfn_t<N> d = 0;
    d.set(n_imp - 1 - imp_up_orb);
    d.set(n_active - 1 - bath_up_p);
    d.set(32 + (n_imp - 1 - imp_dn_orb));
    d.set(32 + (n_active - 1 - bath_dn_p));
    return d;
  };

  std::vector<macis::wfn_t<N>> dets;
  std::vector<double> C;
  for(int iu = 0; iu < 2; ++iu)
    for(int bu = 0; bu < 2; ++bu)
      for(int id = 0; id < 2; ++id)
        for(int bd = 0; bd < 2; ++bd) {
          dets.push_back(make_det(iu, bu, id, bd));
          const double c = 0.05 * (double)(dets.size()) - 0.31;
          C.push_back(c);
        }
  double norm = 0.0;
  for(double c : C) norm += c * c;
  norm = std::sqrt(norm);
  for(double& c : C) c /= norm;

  std::vector<double> orb_rot(n_active * n_active, 0.0);
  orb_rot[0 * n_active + 0] = 0.6;
  orb_rot[0 * n_active + 1] = 0.8;
  orb_rot[1 * n_active + 0] = -0.8;
  orb_rot[1 * n_active + 1] = 0.6;
  orb_rot[2 * n_active + 2] = 1.0;
  orb_rot[3 * n_active + 3] = 1.0;

  auto rho_dense = macis::compute_impurity_rdm_from_state<N>(
      n_imp, n_active, dets, C, orb_rot);
  auto rdm_blocks = macis::compute_impurity_rdm_blocks_from_state<N>(
      n_imp, n_active, dets, C, orb_rot);
  auto rho_sparse = rdm_blocks.to_dense();

  double max_diff = 0.0;
  for(size_t i = 0; i < dim * dim; ++i)
    max_diff = std::max(max_diff, std::abs(rho_dense[i] - rho_sparse[i]));

  SECTION("block-sparse RDM matches dense reference") {
    REQUIRE(max_diff < 1e-12);
  }

  SECTION("block structure is consistent") {
    REQUIRE(rdm_blocks.basis_size == dim);
    for(size_t b = 0; b < rdm_blocks.blocks.size(); ++b) {
      const auto& blk = rdm_blocks.blocks[b];
      for(size_t r = 0; r < blk.dim; ++r) {
        const size_t i = blk.basis_indices[r];
        const uint64_t s = (uint64_t)(dim - 1 - i);
        const int n_up =
            (int)std::bitset<64>((s >> n_imp) & mask).count();
        const int n_dn = (int)std::bitset<64>(s & mask).count();
        REQUIRE(n_up == blk.n_up);
        REQUIRE(n_dn == blk.n_dn);
        REQUIRE(rdm_blocks.block_of[i] == b);
        REQUIRE(rdm_blocks.local_of[i] == r);
      }
    }
  }

  SECTION("trace equals 1: Tr(rho) == 1") {
    REQUIRE(rdm_blocks.trace() == Approx(1.0).epsilon(1e-10));
  }

  SECTION("dense binary writer round-trips") {
    rdm_blocks.write_dense_bin("imp_rdm_test_dense.bin");
    std::ifstream in("imp_rdm_test_dense.bin", std::ios::binary);
    int bs = 0;
    in.read(reinterpret_cast<char*>(&bs), sizeof(int));
    REQUIRE(bs == (int)dim);
    std::vector<double> back(dim * dim, 0.0);
    in.read(reinterpret_cast<char*>(back.data()),
            (std::streamsize)(back.size() * sizeof(double)));
    REQUIRE(back == rho_sparse);
  }

  SECTION("block and text writers produce output") {
    rdm_blocks.write_blocks_bin("imp_rdm_test_blocks.bin", 0.0);
    rdm_blocks.write_dense_txt("imp_rdm_test.txt");
    std::ifstream b("imp_rdm_test_blocks.bin", std::ios::binary | std::ios::ate);
    std::ifstream t("imp_rdm_test.txt");
    REQUIRE(b.good());
    REQUIRE(b.tellg() > 0);
    REQUIRE(t.good());
  }

  SECTION("mixed impurity particle-number sectors throw") {
    macis::wfn_t<N> d_a = 0;
    d_a.set(0);
    d_a.set(1);
    d_a.set(3);
    d_a.set(32 + 1);
    d_a.set(32 + 3);
    macis::wfn_t<N> d_b = 0;
    d_b.set(1);
    d_b.set(3);
    d_b.set(32 + 1);
    d_b.set(32 + 3);
    std::vector<macis::wfn_t<N>> mixed = {d_a, d_b};
    std::vector<double> Cm = {0.6, 0.8};
    REQUIRE_THROWS(macis::compute_impurity_rdm_blocks_from_state<N>(
        n_imp, n_active, mixed, Cm, orb_rot));
  }
}

TEST_CASE("Impurity RDM - block-sparse matches dense across sectors") {
  ROOT_ONLY(MPI_COMM_WORLD);

  // Every determinant with N_up = 2, N_dn = 1 on 2 impurity + 2 bath
  // orbitals: the impurity spans n_up in {0,1,2} and n_dn in {0,1}, so several
  // (N_up,N_dn) blocks are filled and the fermionic sign varies with the bath
  // filling -- unlike the single-block test above.
  constexpr size_t N = 64;
  const size_t n_imp = 2;
  const size_t n_active = 4;
  const size_t dim = 1 << (2 * n_imp);

  std::vector<macis::wfn_t<N>> dets;
  std::vector<double> C;
  for(uint64_t up = 0; up < 16; ++up) {
    if(std::bitset<64>(up).count() != 2) continue;
    for(uint64_t dn = 0; dn < 16; ++dn) {
      if(std::bitset<64>(dn).count() != 1) continue;
      dets.push_back(macis::wfn_t<N>(up | (dn << 32)));
      C.push_back(std::sin(1.7 * (double)dets.size() + 0.3));
    }
  }
  REQUIRE(dets.size() == 24);
  double norm = 0.0;
  for(double c : C) norm += c * c;
  for(double& c : C) c /= std::sqrt(norm);

  // Block-diagonal in imp/bath, as rotate_hamiltonian_ordm_imp_bath produces.
  std::vector<double> orb_rot(n_active * n_active, 0.0);
  orb_rot[0 * n_active + 0] = 0.6;
  orb_rot[0 * n_active + 1] = 0.8;
  orb_rot[1 * n_active + 0] = -0.8;
  orb_rot[1 * n_active + 1] = 0.6;
  orb_rot[2 * n_active + 2] = 0.28;
  orb_rot[2 * n_active + 3] = 0.96;
  orb_rot[3 * n_active + 2] = -0.96;
  orb_rot[3 * n_active + 3] = 0.28;

  auto rho_dense = macis::compute_impurity_rdm_from_state<N>(
      n_imp, n_active, dets, C, orb_rot);
  auto rdm_blocks = macis::compute_impurity_rdm_blocks_from_state<N>(
      n_imp, n_active, dets, C, orb_rot);
  auto rho_sparse = rdm_blocks.to_dense();

  double max_diff = 0.0;
  for(size_t i = 0; i < dim * dim; ++i)
    max_diff = std::max(max_diff, std::abs(rho_dense[i] - rho_sparse[i]));

  size_t filled = 0;
  double tr_sum = 0.0;
  for(const auto& blk : rdm_blocks.blocks) {
    double tr = 0.0;
    for(size_t r = 0; r < blk.dim; ++r) tr += blk.matrix[r * blk.dim + r];
    if(tr > 1e-12) ++filled;
    tr_sum += tr;
  }

  SECTION("several blocks are filled") { REQUIRE(filled >= 4); }

  SECTION("block-sparse RDM matches dense reference") {
    REQUIRE(max_diff < 1e-12);
  }

  SECTION("no weight outside the blocks in the dense reference") {
    for(size_t i = 0; i < dim; ++i)
      for(size_t j = 0; j < dim; ++j)
        if(rdm_blocks.block_of[i] != rdm_blocks.block_of[j])
          REQUIRE(std::abs(rho_dense[i * dim + j]) < 1e-14);
  }

  SECTION("block traces sum to 1") { REQUIRE(tr_sum == Approx(1.0)); }
}
