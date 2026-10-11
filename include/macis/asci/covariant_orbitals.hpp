/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <lapack.hh>
#pragma GCC diagnostic pop

namespace macis {

/**
 *  @brief The orbital blocks whose 1-RDM is diagonalized separately for the
 *  impurity/bath natural orbitals: the impurity [0, nimps) and the bath
 *  [nimps, norb), each split by group_of when it is given. An orbital of
 *  group -1 (in no group) is a block of its own. Every index is in exactly
 *  one block, ascending inside it.
 *
 *  Shared by rotate_hamiltonian_ordm_imp_bath and covariant_natural_orbitals,
 *  so the two rotations cannot disagree about what a block is.
 */
inline std::vector<std::vector<size_t>> natural_orbital_blocks(
    size_t norb, size_t nimps, const std::vector<int>* group_of) {
  if(group_of != nullptr and group_of->size() != norb)
    throw std::runtime_error("natural_orbital_blocks: group_of has " +
                             std::to_string(group_of->size()) +
                             " entries, expected " + std::to_string(norb));
  std::vector<std::vector<size_t>> blocks;
  for(int side = 0; side < 2; ++side) {
    const size_t lo = side ? nimps : 0, hi = side ? norb : nimps;
    if(group_of == nullptr) {
      blocks.emplace_back();
      for(size_t i = lo; i < hi; ++i) blocks.back().push_back(i);
      continue;
    }
    std::map<int, std::vector<size_t>> by_group;
    for(size_t i = lo; i < hi; ++i) {
      const int g = (*group_of)[i];
      if(g < 0)
        blocks.push_back({i});
      else
        by_group[g].push_back(i);
    }
    for(auto& [g, idx] : by_group) blocks.push_back(std::move(idx));
  }
  return blocks;
}

/**
 *  @brief Group-covariant natural orbitals of a 1-RDM
 *  (PLAN_symmetric_asci_search.md, §12.2).
 *
 *  Returns the block-diagonal rotation U (column major, columns = natural
 *  orbitals) and the occupations, such that for every g in @p group
 *
 *      U[g(p), g(q)] == U[p, q]    and    occs[g(p)] == occs[p],
 *
 *  bit for bit. The permutation group then still acts as a pure orbital
 *  relabeling in the rotated basis, which is what SYMMETRIZE_DETS needs.
 *  Independent per-block diagonalizations do not give this: a degenerate
 *  occupation subspace, the sign of each vector and the order of the vectors
 *  are all arbitrary, and differ between partner blocks.
 *
 *   1. ordm is averaged over the group, rbar[p,q] = 1/|G| sum_g ordm[g(p),g(q)].
 *      That only picks the basis; the state is untouched. For a symmetric state
 *      rbar = ordm up to the solver's tolerance.
 *   2. Every g must map every block onto a block (throws otherwise: the
 *      blocks and the group disagree).
 *   3. Every g that maps a block onto itself must fix each of its orbitals
 *      (throws otherwise: a symmetry-adapted diagonalization inside the block
 *      would be needed, which is not implemented). Band permutations always
 *      pass; site permutations inside a band block do not.
 *   4. The lowest-indexed block of each block orbit is diagonalized (gesvd, as
 *      in rotate_hamiltonian_ordm_imp_bath: occupations descending), and its
 *      vectors are copied onto the other blocks of the orbit by any g that
 *      maps it there. By step 3 the choice of g does not matter.
 *
 *  @param[in]  ordm     norb x norb 1-RDM, column major
 *  @param[in]  blocks   natural_orbital_blocks(...): a partition of [0, norb)
 *  @param[in]  group    the full permutation group (identity optional)
 *  @param[out] U        norb x norb rotation
 *  @param[out] occs_out norb natural-orbital occupations, or nullptr
 *  @returns max |ordm - rbar|, how far the state's 1-RDM is from symmetric
 */
inline double covariant_natural_orbitals(
    const double* ordm, size_t norb,
    const std::vector<std::vector<size_t>>& blocks,
    const std::vector<std::vector<uint32_t>>& group, double* U,
    double* occs_out) {
  for(const auto& g : group) {
    bool ok = g.size() == norb;
    std::vector<bool> seen(norb, false);
    for(size_t p = 0; p < g.size() and ok; ++p) {
      ok = g[p] < norb and !seen[g[p]];
      if(ok) seen[g[p]] = true;
    }
    if(!ok)
      throw std::runtime_error(
          "covariant_natural_orbitals: a group element is not a permutation "
          "of the " +
          std::to_string(norb) + " active orbitals");
  }

  std::vector<int> block_of(norb, -1);
  for(size_t b = 0; b < blocks.size(); ++b)
    for(auto p : blocks[b]) {
      if(p >= norb or block_of[p] >= 0)
        throw std::runtime_error(
            "covariant_natural_orbitals: the blocks are not a partition of "
            "the active orbitals");
      block_of[p] = int(b);
    }
  for(size_t p = 0; p < norb; ++p)
    if(block_of[p] < 0)
      throw std::runtime_error(
          "covariant_natural_orbitals: orbital " + std::to_string(p) +
          " is in no block");

  // 1. Group average of the 1-RDM
  std::vector<double> rbar(norb * norb, 0.);
  if(group.empty())
    std::copy(ordm, ordm + norb * norb, rbar.begin());
  else {
    for(const auto& g : group)
      for(size_t q = 0; q < norb; ++q)
        for(size_t p = 0; p < norb; ++p)
          rbar[p + q * norb] += ordm[g[p] + g[q] * norb];
    for(auto& x : rbar) x /= double(group.size());
  }
  double asym = 0.;
  for(size_t i = 0; i < norb * norb; ++i)
    asym = std::max(asym, std::abs(ordm[i] - rbar[i]));

  // 2.-3. Block images and stabilizers
  // image[b][ig] = the block g_ig maps block b onto
  std::vector<std::vector<int>> image(blocks.size(),
                                      std::vector<int>(group.size(), -1));
  for(size_t b = 0; b < blocks.size(); ++b)
    for(size_t ig = 0; ig < group.size(); ++ig) {
      const auto& g = group[ig];
      const auto& A = blocks[b];
      if(A.empty()) continue;
      const int c = block_of[g[A.front()]];
      for(auto p : A)
        if(block_of[g[p]] != c)
          throw std::runtime_error(
              "covariant_natural_orbitals: a permutation of the symmetry group "
              "splits the natural-orbital block of orbital " +
              std::to_string(A.front()) +
              " across blocks: the band partition and the group disagree");
      if(blocks[c].size() != A.size())
        throw std::runtime_error(
            "covariant_natural_orbitals: a permutation maps the "
            "natural-orbital block of orbital " +
            std::to_string(A.front()) + " onto a block of a different size");
      if(c == int(b))
        for(auto p : A)
          if(g[p] != p)
            throw std::runtime_error(
                "covariant_natural_orbitals: a permutation of the symmetry "
                "group maps the natural-orbital block of orbital " +
                std::to_string(A.front()) +
                " onto itself but moves orbital " + std::to_string(p) +
                " inside it. Only groups that permute whole blocks (e.g. band "
                "swaps) are supported with NROTS > 0; a permutation inside a "
                "block would need a symmetry-adapted diagonalization");
      image[b][ig] = c;
    }

  // 4. Diagonalize one block per orbit, transport to the rest
  std::fill(U, U + norb * norb, 0.);
  std::vector<bool> done(blocks.size(), false);
  for(size_t a = 0; a < blocks.size(); ++a) {
    if(done[a] or blocks[a].empty()) continue;
    const auto& A = blocks[a];
    const size_t m = A.size();
    std::vector<double> nat(m * m), S(m);
    for(size_t j = 0; j < m; ++j)
      for(size_t i = 0; i < m; ++i) nat[i + j * m] = rbar[A[i] + A[j] * norb];
    lapack::gesvd(lapack::Job::OverwriteVec, lapack::Job::NoVec, m, m,
                  nat.data(), m, S.data(), NULL, 1, NULL, 1);

    // The representative itself, then one copy per other block of the orbit
    auto place = [&](const std::vector<uint32_t>* g) {
      auto img = [&](size_t p) { return g ? size_t((*g)[p]) : p; };
      for(size_t j = 0; j < m; ++j) {
        if(occs_out) occs_out[img(A[j])] = S[j];
        for(size_t i = 0; i < m; ++i)
          U[img(A[i]) + img(A[j]) * norb] = nat[i + j * m];
      }
    };
    place(nullptr);
    done[a] = true;
    for(size_t ig = 0; ig < group.size(); ++ig) {
      const int c = image[a][ig];
      if(done[c]) continue;
      place(&group[ig]);
      done[c] = true;
    }
  }
  return asym;
}

/**
 *  @brief Largest |U[g(p), g(q)] - U[p, q]| over the group: zero for a
 *  G-covariant rotation (condition (★) of PLAN_symmetric_asci_search.md
 *  §12).
 */
inline double max_covariance_violation(
    const double* U, size_t norb,
    const std::vector<std::vector<uint32_t>>& group) {
  double m = 0.;
  for(const auto& g : group) {
    if(g.size() != norb)
      throw std::runtime_error(
          "max_covariance_violation: a permutation has " +
          std::to_string(g.size()) + " entries, expected " +
          std::to_string(norb));
    for(size_t q = 0; q < norb; ++q)
      for(size_t p = 0; p < norb; ++p)
        m = std::max(m, std::abs(U[g[p] + g[q] * norb] - U[p + q * norb]));
  }
  return m;
}

}  // namespace macis
