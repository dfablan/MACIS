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
#include <limits>
#include <macis/asci/parity_labels.hpp>
#include <macis/bitset_operations.hpp>
#include <macis/hamiltonian_generator.hpp>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace macis {

/**
 *  @brief Detect, verify and clean the band-parity labels of an impurity
 *  Hamiltonian (ASCI.PARITY_SOLVE). See parity-sector-solve-simple.md, §2.
 *
 *  Impurity orbital i < n_imp belongs to band i / (n_imp / nbands) (band-major
 *  layout). A bath orbital belongs to the band it is connected to through
 *  one-body couplings |T_pq| > tol (|Td_pq| too if Td is given); one connected
 *  to two bands means the bath is not band-diagonal, and this throws. An
 *  orbital connected to no impurity orbital is "decoupled" (group -1).
 *
 *  Every one-body element between different groups and every two-body
 *  element (pq|rs) whose four indices put an odd count into some group (or
 *  some decoupled orbital) breaks a parity: those with |x| <= tol are set to
 *  exactly 0 in T, Td and V (the largest is reported as max_discarded); any
 *  larger one throws. The parity test sums the memberships of all four
 *  indices, so it does not depend on the integral index convention.
 *
 *  @param[in,out] T   norb x norb one-body integrals (column major)
 *  @param[in,out] Td  spin-down one-body integrals, or nullptr
 *  @param[in,out] V   norb^4 two-body integrals, V[p + q n + r n^2 + s n^3] =
 *                     (pq|rs)
 *  @param[out]    log where the PARITY_LABELS summary and warnings go
 */
ParityLabels build_parity_labels(size_t norb, size_t n_imp, size_t nbands,
                                 std::vector<double>& T,
                                 std::vector<double>* Td,
                                 std::vector<double>& V, double tol,
                                 std::ostream& log);

/**
 *  @brief All parity keys a determinant with `nel_grouped` electrons in the
 *  grouped (non-decoupled) orbitals can have: 2^(ngroups-1) of them.
 */
inline std::vector<uint32_t> enumerate_parity_keys(const ParityLabels& L,
                                                   size_t nel_grouped) {
  if(L.ngroups == 0 or L.ngroups > 31)
    throw std::runtime_error("enumerate_parity_keys: need 1..31 groups, got " +
                             std::to_string(L.ngroups));
  std::vector<uint32_t> keys;
  for(uint32_t k = 0; k < (1u << L.ngroups); ++k) {
    size_t pop = 0;
    for(size_t g = 0; g < L.ngroups; ++g) pop += (k >> g) & 1u;
    if(pop % 2 == nel_grouped % 2) keys.push_back(k);
  }
  return keys;
}

/**
 *  @brief The elements of an (expanded) orbital-permutation group that keep
 *  the parity sector `key`: g must map every group onto a group with the same
 *  parity bit, and decoupled orbitals onto decoupled orbitals. These form a
 *  subgroup, so no re-expansion is needed. Throws if some element maps a
 *  group onto a set that is not a group: such a g is not a symmetry of a
 *  parity-labelled Hamiltonian.
 */
inline std::vector<std::vector<uint32_t>> parity_stabilizer(
    const std::vector<std::vector<uint32_t>>& group, const ParityLabels& L,
    uint32_t key) {
  std::vector<std::vector<uint32_t>> out;
  for(const auto& g : group) {
    bool keep = true;
    for(size_t q = 0; q < L.group_of.size() and keep; ++q)
      if(L.group_of[q] < 0) keep = L.group_of[g[q]] < 0;
    for(size_t a = 0; a < L.ngroups and keep; ++a) {
      const auto& orbs = L.group_orbs[a];
      const int b = L.group_of[g[orbs.front()]];
      if(b < 0) {
        keep = false;
        break;
      }
      for(auto q : orbs)
        if(L.group_of[g[q]] != b)
          throw std::runtime_error(
              "parity_stabilizer: a permutation of the symmetry group splits "
              "band group " +
              std::to_string(a) + " across groups");
      if(L.group_orbs[b].size() != orbs.size())
        throw std::runtime_error(
            "parity_stabilizer: a permutation maps band group " +
            std::to_string(a) + " onto a group of a different size");
      keep = ((key >> a) & 1u) == ((key >> b) & 1u);
    }
    if(keep) out.push_back(g);
  }
  return out;
}

namespace detail {

// One electron of one spin moved from orbital i to orbital a
struct ParityMove {
  uint32_t i, a;
  bool beta;
};

template <size_t N>
wfn_t<N> apply_move(wfn_t<N> d, const ParityMove& m) {
  const size_t off = m.beta ? N / 2 : 0;
  d.reset(m.i + off);
  d.set(m.a + off);
  return d;
}

// All single moves of d among the grouped orbitals
template <size_t N>
std::vector<ParityMove> grouped_singles(const wfn_t<N>& d,
                                        const ParityLabels& L) {
  std::vector<ParityMove> moves;
  const size_t norb = L.group_of.size();
  for(int s = 0; s < 2; ++s) {
    const size_t off = s ? N / 2 : 0;
    for(uint32_t i = 0; i < norb; ++i) {
      if(L.group_of[i] < 0 or !d[i + off]) continue;
      for(uint32_t a = 0; a < norb; ++a) {
        if(L.group_of[a] < 0 or d[a + off]) continue;
        moves.push_back({i, a, bool(s)});
      }
    }
  }
  return moves;
}

}  // namespace detail

/**
 *  @brief Starting determinant for the parity sector `target`
 *  (parity-sector-solve-simple.md, §4).
 *
 *  If `base` is already in the sector it is returned unchanged, so the sector
 *  today's solver reaches runs exactly as before. Otherwise:
 *   1. repair: while the key differs, apply the single move between two
 *      groups whose parities are both wrong that gives the lowest diagonal
 *      energy <D|H|D>;
 *   2. descend: apply the sector-preserving move that lowers <D|H|D> the
 *      most -- any single move, or a double move whose two halves each touch
 *      an impurity orbital (the interaction lives there; bath rearrangements
 *      are one-body and reached by singles) -- until none lowers it, at most
 *      max_steps times.
 *  Decoupled orbitals are never touched. The diagonal energy includes U, U'
 *  and J, which is what makes the seed sensible at large U, where one-body
 *  order is a poor guide.
 */
template <size_t N>
wfn_t<N> parity_seed(const wfn_t<N>& base, uint32_t target,
                     const ParityLabels& L, HamiltonianGenerator<N>& H,
                     size_t n_imp, size_t max_steps = 50) {
  const ParityMasks<N> pm(L);
  if(pm.key(base) == target) return base;

  wfn_t<N> D = base;
  auto Ediag = [&](const wfn_t<N>& d) { return H.matrix_element(d, d); };

  // 1. Repair
  while(pm.key(D) != target) {
    const uint32_t wrong = pm.key(D) ^ target;
    double best = std::numeric_limits<double>::infinity();
    wfn_t<N> best_d;
    for(const auto& m : detail::grouped_singles(D, L)) {
      const int gi = L.group_of[m.i], ga = L.group_of[m.a];
      if(gi == ga or !((wrong >> gi) & 1u) or !((wrong >> ga) & 1u)) continue;
      const auto d = detail::apply_move(D, m);
      const double e = Ediag(d);
      if(e < best) best = e, best_d = d;
    }
    if(!std::isfinite(best))
      throw std::runtime_error(
          "parity_seed: no single move reaches sector " +
          parity_key_string(target, L.ngroups) + " from " +
          parity_key_string(pm.key(D), L.ngroups) +
          " (a band group is completely full or completely empty)");
    D = best_d;
  }

  // 2. Descend
  auto touches_imp = [&](const detail::ParityMove& m) {
    return m.i < n_imp or m.a < n_imp;
  };
  double E = Ediag(D);
  for(size_t step = 0; step < max_steps; ++step) {
    double best = E;
    wfn_t<N> best_d = D;
    for(const auto& m1 : detail::grouped_singles(D, L)) {
      const auto d1 = detail::apply_move(D, m1);
      if(pm.key(d1) == target) {
        const double e = Ediag(d1);
        if(e < best - 1e-12) best = e, best_d = d1;
      }
      if(!touches_imp(m1)) continue;
      for(const auto& m2 : detail::grouped_singles(d1, L)) {
        if(!touches_imp(m2)) continue;
        const auto d2 = detail::apply_move(d1, m2);
        if(d2 == D or pm.key(d2) != target) continue;
        const double e = Ediag(d2);
        if(e < best - 1e-12) best = e, best_d = d2;
      }
    }
    if(best_d == D) break;
    D = best_d;
    E = best;
  }
  return D;
}

}  // namespace macis
