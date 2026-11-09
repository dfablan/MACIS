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
 *  @brief A signed orbital permutation, the form a cluster translation takes
 *  on impurity sites plus a momentum-resolved (Irrep) bath (ASCI.ROT_ADAPT):
 *
 *      phi_p  ->  sign[p] * phi_{perm[p]},   i.e.  M[perm[p], p] = sign[p].
 *
 *  It is a symmetry of H when, for all indices,
 *      T[perm p, perm q] s_p s_q = T[p, q]   (and the same with four signs
 *  for V). The impurity sites are permuted, a bath orbital coupled to one
 *  cluster momentum k only picks up the character of k. The sign of a whole
 *  band (identity permutation, sign -1 on the band's orbitals) is also one:
 *  its character on a determinant is that band's parity.
 */
struct SignedPerm {
  std::vector<uint32_t> perm;
  std::vector<int> sign;
};

/// Dense column-major matrix of a signed permutation, M[perm[p] + p n] = s_p
inline std::vector<double> signed_perm_matrix(const SignedPerm& g, size_t n) {
  if(g.perm.size() != n or g.sign.size() != n)
    throw std::runtime_error("signed_perm_matrix: generator has " +
                             std::to_string(g.perm.size()) + " / " +
                             std::to_string(g.sign.size()) +
                             " entries, expected " + std::to_string(n));
  std::vector<double> M(n * n, 0.);
  for(size_t p = 0; p < n; ++p) M[g.perm[p] + p * n] = double(g.sign[p]);
  return M;
}

/// R^T M R for column-major n x n matrices: an orbital operator expressed in
/// the basis whose orbitals are the columns of R (R = orb_rot)
inline std::vector<double> to_rotated_basis(const std::vector<double>& M,
                                            const double* R, size_t n) {
  std::vector<double> tmp(n * n, 0.), out(n * n, 0.);
  for(size_t j = 0; j < n; ++j)
    for(size_t k = 0; k < n; ++k) {
      const double r = R[k + j * n];
      if(r == 0.) continue;
      for(size_t i = 0; i < n; ++i) tmp[i + j * n] += M[i + k * n] * r;
    }
  for(size_t j = 0; j < n; ++j)
    for(size_t i = 0; i < n; ++i) {
      double s = 0.;
      for(size_t k = 0; k < n; ++k) s += R[k + i * n] * tmp[k + j * n];
      out[i + j * n] = s;
    }
  return out;
}

/**
 *  @brief Symmetry-adapted natural orbitals of one block (ASCI.ROT_ADAPT).
 *
 *  ops are commuting involutions (signed permutations in the current
 *  orbital basis) that map the block onto itself. The block is first split
 *  into their joint eigenspaces ("character classes": the +-1 pattern of
 *  every op), then rbar is diagonalized inside each class. Occupations are
 *  returned in descending order over the whole block, like gesvd.
 *
 *  Why: plain diagonalization leaves any degenerate occupation subspace in
 *  an arbitrary basis. On a C4-symmetric 2x2 cluster the X and Y orbitals
 *  are degenerate, so the natural orbitals came out as X/Y mixtures (36 deg
 *  in J0.333/U15 It_9). Total momentum is then not a quantum number of the
 *  determinants, and operators at q != 0 stay H-connected to the ground
 *  state's sector. With adapted orbitals every determinant has a definite
 *  character under each op, and H conserves it.
 *
 *  The class split equals averaging rbar over the abelian group the ops
 *  generate (basis choice only, as for covariant_natural_orbitals' rbar).
 *
 *  @param[in]  rbar  norb x norb 1-RDM (column major, current basis)
 *  @param[in]  A     the block's orbital indices
 *  @param[in]  ops   norb x norb column-major operator matrices
 *  @param[out] nat   m x m eigenvectors (column major, columns = orbitals)
 *  @param[out] S     m occupations, descending
 *  @returns max |rbar| element between different classes (0 for a
 *           symmetric state)
 */
inline double symmetry_adapted_block_orbitals(
    const std::vector<double>& rbar, size_t norb, const std::vector<size_t>& A,
    const std::vector<std::vector<double>>& ops, std::vector<double>& nat,
    std::vector<double>& S) {
  const size_t m = A.size();
  const double tol = 1e-8;
  if(ops.size() > 30)
    throw std::runtime_error(
        "symmetry_adapted_block_orbitals: more than 30 generators");

  // Restrictions of the ops to the block, checked closed and involutive
  std::vector<std::vector<double>> R(ops.size(), std::vector<double>(m * m));
  std::vector<bool> in_block(norb, false);
  for(auto p : A) in_block[p] = true;
  for(size_t g = 0; g < ops.size(); ++g) {
    const auto& M = ops[g];
    for(size_t j = 0; j < m; ++j) {
      for(size_t p = 0; p < norb; ++p)
        if(!in_block[p] and std::abs(M[p + A[j] * norb]) > tol)
          throw std::runtime_error(
              "ASCI.ROT_ADAPT: generator " + std::to_string(g + 1) +
              " maps orbital " + std::to_string(A[j]) +
              " out of its natural-orbital block (blocks are impurity / bath "
              "per band group): the generators and the blocks disagree");
      for(size_t i = 0; i < m; ++i) R[g][i + j * m] = M[A[i] + A[j] * norb];
    }
    for(size_t j = 0; j < m; ++j)
      for(size_t i = 0; i < m; ++i) {
        double sq = 0.;
        for(size_t k = 0; k < m; ++k) sq += R[g][i + k * m] * R[g][k + j * m];
        if(std::abs(sq - (i == j ? 1. : 0.)) > tol or
           std::abs(R[g][i + j * m] - R[g][j + i * m]) > tol)
          throw std::runtime_error(
              "ASCI.ROT_ADAPT: generator " + std::to_string(g + 1) +
              " is not a symmetric involution on the block of orbital " +
              std::to_string(A.front()) +
              " (only Z2 generators are supported: real orbitals cannot carry "
              "complex characters)");
      }
  }

  // Joint eigenbasis: C = sum_g 2^-g R_g has a distinct eigenvalue for every
  // +-1 pattern, separated by at least 2^(1-G)
  std::vector<double> C(m * m, 0.), w(m);
  for(size_t g = 0; g < ops.size(); ++g)
    for(size_t k = 0; k < m * m; ++k) C[k] += std::ldexp(R[g][k], -int(g));
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, m, C.data(), m, w.data());

  // Character class of each joint eigenvector
  std::map<uint32_t, std::vector<size_t>> classes;
  for(size_t i = 0; i < m; ++i) {
    uint32_t key = 0;
    for(size_t g = 0; g < ops.size(); ++g) {
      double chi = 0.;
      for(size_t b = 0; b < m; ++b)
        for(size_t a = 0; a < m; ++a)
          chi += C[a + i * m] * R[g][a + b * m] * C[b + i * m];
      if(std::abs(std::abs(chi) - 1.) > 1e-6)
        throw std::runtime_error(
            "ASCI.ROT_ADAPT: the generators do not commute on the block of "
            "orbital " +
            std::to_string(A.front()) + " (character " + std::to_string(chi) +
            ")");
      if(chi < 0.) key |= (1u << g);
    }
    classes[key].push_back(i);
  }

  // rbar restricted to the block, then diagonalized inside each class
  std::vector<double> rb(m * m);
  for(size_t j = 0; j < m; ++j)
    for(size_t i = 0; i < m; ++i) rb[i + j * m] = rbar[A[i] + A[j] * norb];
  auto project = [&](const std::vector<size_t>& ci,
                     const std::vector<size_t>& cj) {
    std::vector<double> out(ci.size() * cj.size(), 0.);
    for(size_t b = 0; b < cj.size(); ++b)
      for(size_t a = 0; a < ci.size(); ++a) {
        double s = 0.;
        for(size_t l = 0; l < m; ++l)
          for(size_t k = 0; k < m; ++k)
            s += C[k + ci[a] * m] * rb[k + l * m] * C[l + cj[b] * m];
        out[a + b * ci.size()] = s;
      }
    return out;
  };

  double leak = 0.;
  for(auto it = classes.begin(); it != classes.end(); ++it)
    for(auto jt = std::next(it); jt != classes.end(); ++jt)
      for(double x : project(it->second, jt->second))
        leak = std::max(leak, std::abs(x));

  struct Orbital {
    double occ;
    uint32_t key;
    std::vector<double> vec;
  };
  std::vector<Orbital> orbs;
  for(const auto& [key, ci] : classes) {
    const size_t d = ci.size();
    auto rc = project(ci, ci);
    std::vector<double> ev(d);
    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, d, rc.data(), d,
                 ev.data());
    for(size_t a = 0; a < d; ++a) {
      Orbital o{ev[a], key, std::vector<double>(m, 0.)};
      for(size_t b = 0; b < d; ++b)
        for(size_t k = 0; k < m; ++k)
          o.vec[k] += C[k + ci[b] * m] * rc[b + a * d];
      orbs.push_back(std::move(o));
    }
  }
  // Descending occupations, ties by class: deterministic
  std::stable_sort(orbs.begin(), orbs.end(),
                   [](const Orbital& x, const Orbital& y) {
                     return x.occ != y.occ ? x.occ > y.occ : x.key < y.key;
                   });
  nat.assign(m * m, 0.);
  S.assign(m, 0.);
  for(size_t j = 0; j < m; ++j) {
    S[j] = orbs[j].occ;
    for(size_t i = 0; i < m; ++i) nat[i + j * m] = orbs[j].vec[i];
  }
  return leak;
}

/**
 *  @brief Characters of the rotated orbitals under signed-permutation
 *  generators: bit g of mask[p] is set when orbital p (column p of R) is odd
 *  under generator g. Returns the largest deviation of R^T M_g R from a
 *  diagonal +-1 matrix (0 when every rotated orbital is a joint
 *  eigenvector).
 */
inline double orbital_character_masks(
    const double* R, size_t n, const std::vector<std::vector<double>>& gens,
    std::vector<uint64_t>& mask) {
  mask.assign(n, 0);
  double dev = 0.;
  for(size_t g = 0; g < gens.size(); ++g) {
    const auto Mr = to_rotated_basis(gens[g], R, n);
    for(size_t q = 0; q < n; ++q)
      for(size_t p = 0; p < n; ++p) {
        if(p == q) {
          dev = std::max(dev, std::abs(std::abs(Mr[p + p * n]) - 1.));
          if(Mr[p + p * n] < 0.) mask[p] |= (uint64_t(1) << g);
        } else
          dev = std::max(dev, std::abs(Mr[p + q * n]));
      }
  }
  return dev;
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
 *   1. ordm is averaged over the group, rbar[p,q] = 1/|G| sum_g
 * ordm[g(p),g(q)]. That only picks the basis; the state is untouched. For a
 * symmetric state rbar = ordm up to the solver's tolerance.
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
 *   5. Orbitals [0, nfixed) are kept fixed (ASCI.ROT_IMP_FIXED): every block
 *      inside that range gets U = 1 and the occupations diag(rbar), and the
 *      range must be a union of blocks mapped onto each other by the group.
 *      The identity is trivially covariant, so (★) still holds.
 *
 *  @param[in]  ordm     norb x norb 1-RDM, column major
 *  @param[in]  blocks   natural_orbital_blocks(...): a partition of [0, norb)
 *  @param[in]  group    the full permutation group (identity optional)
 *  @param[out] U        norb x norb rotation
 *  @param[out] occs_out norb natural-orbital occupations, or nullptr
 *  @param[in]  nfixed   orbitals [0, nfixed) are not rotated (default 0)
 *  @param[in]  adapt    ASCI.ROT_ADAPT: commuting Z2 orbital operators in the
 *                       current basis (column major, norb x norb). When given,
 *                       the representative block is diagonalized class by
 *                       class (symmetry_adapted_block_orbitals) instead of
 *                       by gesvd. They must commute with the group (checked
 *                       at solver entry), so the transported copies are
 *                       adapted too.
 *  @param[out] adapt_leak max |rbar| between character classes, or nullptr
 *  @returns max |ordm - rbar|, how far the state's 1-RDM is from symmetric
 */
inline double covariant_natural_orbitals(
    const double* ordm, size_t norb,
    const std::vector<std::vector<size_t>>& blocks,
    const std::vector<std::vector<uint32_t>>& group, double* U,
    double* occs_out, size_t nfixed = 0,
    const std::vector<std::vector<double>>* adapt = nullptr,
    double* adapt_leak = nullptr) {
  if(adapt_leak) *adapt_leak = 0.;
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
      throw std::runtime_error("covariant_natural_orbitals: orbital " +
                               std::to_string(p) + " is in no block");
  auto is_fixed = [&](size_t b) {
    return !blocks[b].empty() and blocks[b].front() < nfixed;
  };
  for(size_t b = 0; b < blocks.size(); ++b)
    for(auto p : blocks[b])
      if((p < nfixed) != is_fixed(b))
        throw std::runtime_error(
            "covariant_natural_orbitals: the fixed range [0, " +
            std::to_string(nfixed) + ") splits the block of orbital " +
            std::to_string(blocks[b].front()));

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
                std::to_string(A.front()) + " onto itself but moves orbital " +
                std::to_string(p) +
                " inside it. Only groups that permute whole blocks (e.g. band "
                "swaps) are supported with NROTS > 0; a permutation inside a "
                "block would need a symmetry-adapted diagonalization");
      if(is_fixed(b) != is_fixed(c))
        throw std::runtime_error(
            "covariant_natural_orbitals: a permutation maps a fixed block "
            "(orbital " +
            std::to_string(A.front()) + ") onto a rotated one");
      image[b][ig] = c;
    }

  // 4. Diagonalize one block per orbit, transport to the rest
  std::fill(U, U + norb * norb, 0.);
  std::vector<bool> done(blocks.size(), false);
  for(size_t a = 0; a < blocks.size(); ++a) {
    if(done[a] or blocks[a].empty()) continue;
    const auto& A = blocks[a];
    const size_t m = A.size();
    std::vector<double> nat(m * m, 0.), S(m);
    if(is_fixed(a))
      for(size_t j = 0; j < m; ++j) {
        nat[j + j * m] = 1.;
        S[j] = rbar[A[j] + A[j] * norb];
      }
    else if(adapt and !adapt->empty()) {
      const double leak =
          symmetry_adapted_block_orbitals(rbar, norb, A, *adapt, nat, S);
      if(adapt_leak) *adapt_leak = std::max(*adapt_leak, leak);
    } else {
      for(size_t j = 0; j < m; ++j)
        for(size_t i = 0; i < m; ++i) nat[i + j * m] = rbar[A[i] + A[j] * norb];
      lapack::gesvd(lapack::Job::OverwriteVec, lapack::Job::NoVec, m, m,
                    nat.data(), m, S.data(), NULL, 1, NULL, 1);
    }

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
      throw std::runtime_error("max_covariance_violation: a permutation has " +
                               std::to_string(g.size()) +
                               " entries, expected " + std::to_string(norb));
    for(size_t q = 0; q < norb; ++q)
      for(size_t p = 0; p < norb; ++p)
        m = std::max(m, std::abs(U[g[p] + g[q] * norb] - U[p + q * norb]));
  }
  return m;
}

}  // namespace macis
