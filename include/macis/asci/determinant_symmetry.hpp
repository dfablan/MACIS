/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/bitset_operations.hpp>
#include <macis/types.hpp>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace macis {

/**
 *  @brief Image of a determinant under an orbital permutation.
 *
 *  The permutation acts jointly on both spin sectors (alpha = low N/2 bits,
 *  beta = high N/2 bits): an occupied orbital p maps to perm[p] in the same
 *  spin sector. Orbitals p >= perm.size() are fixed points. Only the
 *  determinant bitstring is relabeled -- no fermionic sign bookkeeping is
 *  needed as long as coefficients are re-solved in the permuted space.
 *
 *  @param[in] w    Determinant bitstring
 *  @param[in] perm perm[p] is the image of orbital p (0-based)
 *  @returns The permuted determinant
 */
template <size_t N>
wfn_t<N> permute_orbitals(wfn_t<N> w, const std::vector<uint32_t>& perm) {
  wfn_t<N> out(0);
  const size_t norb = perm.size();
  for(size_t p = 0; p < norb; ++p) {
    if(w[p]) out.set(perm[p]);
    if(w[p + N / 2]) out.set(perm[p] + N / 2);
  }
  // Pass through untouched orbitals
  for(size_t p = norb; p < N / 2; ++p) {
    if(w[p]) out.set(p);
    if(w[p + N / 2]) out.set(p + N / 2);
  }
  return out;
}

/**
 *  @brief True iff perm has length norb and is a bijection on [0, norb).
 */
inline bool is_valid_permutation(const std::vector<uint32_t>& perm,
                                 size_t norb) {
  if(perm.size() != norb) return false;
  std::vector<bool> seen(norb, false);
  for(auto p : perm) {
    if(p >= norb or seen[p]) return false;
    seen[p] = true;
  }
  return true;
}

/**
 *  @brief Expand a list of permutation generators to the full group.
 *
 *  BFS over composition until closure. Deterministic: the returned elements
 *  are lexicographically sorted and always include the identity.
 *
 *  @param[in] gens     Generator permutations, each of length norb
 *  @param[in] norb     Number of orbitals the permutations act on
 *  @param[in] max_size Throw if the group grows beyond this (misconfigured
 *                      generators); default 8! = 40320
 *  @returns All group elements, lexicographically sorted
 */
inline std::vector<std::vector<uint32_t>> expand_permutation_group(
    const std::vector<std::vector<uint32_t>>& gens, size_t norb,
    size_t max_size = 40320 /* 8! */) {
  for(const auto& g : gens)
    if(!is_valid_permutation(g, norb))
      throw std::runtime_error(
          "expand_permutation_group: generator is not a bijection on [0, " +
          std::to_string(norb) + ")");

  std::vector<uint32_t> id(norb);
  std::iota(id.begin(), id.end(), 0);

  std::set<std::vector<uint32_t>> group{id};
  std::vector<std::vector<uint32_t>> frontier{id};
  while(frontier.size()) {
    std::vector<std::vector<uint32_t>> next;
    for(const auto& e : frontier)
      for(const auto& g : gens) {
        // Composition (g after e): p -> g[e[p]]
        std::vector<uint32_t> c(norb);
        for(size_t p = 0; p < norb; ++p) c[p] = g[e[p]];
        if(group.insert(c).second) {
          if(group.size() > max_size)
            throw std::runtime_error(
                "expand_permutation_group: group exceeds max_size = " +
                std::to_string(max_size) + "; check the input generators");
          next.push_back(std::move(c));
        }
      }
    frontier = std::move(next);
  }

  return std::vector<std::vector<uint32_t>>(group.begin(), group.end());
}

/**
 *  @brief Canonical orbit representative: the minimum over all group images
 *  of a determinant by bitset_less.
 */
template <size_t N>
wfn_t<N> orbit_representative(wfn_t<N> w,
                              const std::vector<std::vector<uint32_t>>& group) {
  wfn_t<N> rep = w;
  for(const auto& g : group) {
    auto img = permute_orbitals(w, g);
    if(bitset_less(img, rep)) rep = img;
  }
  return rep;
}

/**
 *  @brief Whole-orbit budget selection.
 *
 *  Groups the scored determinants into orbits of the permutation group,
 *  scores each orbit by the max |rv| over its members present in the input,
 *  sorts orbits by (score desc, representative asc), and greedily emits whole
 *  orbits while the total stays <= budget. On the first orbit that does not
 *  fit, selection stops (strict budget: later smaller orbits are not
 *  back-filled, so the result is a clean prefix of the orbit ranking).
 *
 *  The result is closed under the group, deterministic (total tie-break by
 *  bitset_less), never exceeds the budget, and is duplicate-free even if the
 *  input list contains duplicate determinants.
 *
 *  @param[in] pairs  Scored determinant list (rank-identical under MPI)
 *  @param[in] group  Full permutation group (identity included)
 *  @param[in] budget Maximum number of determinants to return
 *  @returns G-closed determinant list sorted by bitset_less, size <= budget
 */
template <size_t N>
std::vector<wfn_t<N>> symmetric_orbit_select(
    const asci_contrib_container<wfn_t<N>>& pairs,
    const std::vector<std::vector<uint32_t>>& group, size_t budget) {
  // Aggregate scores by orbit representative
  std::map<wfn_t<N>, double, bitset_less_comparator<N>> orbit_scores;
  for(const auto& p : pairs) {
    auto rep = orbit_representative(p.state, group);
    auto [it, inserted] = orbit_scores.try_emplace(rep, std::abs(p.rv));
    if(!inserted) it->second = std::max(it->second, std::abs(p.rv));
  }

  // Sort orbits by (score desc, representative asc) -- a total order
  std::vector<std::pair<wfn_t<N>, double>> orbits(orbit_scores.begin(),
                                                  orbit_scores.end());
  std::sort(orbits.begin(), orbits.end(), [](const auto& a, const auto& b) {
    if(a.second != b.second) return a.second > b.second;
    return bitset_less(a.first, b.first);
  });

  // Greedily emit whole orbits under a strict budget
  std::vector<wfn_t<N>> selected;
  selected.reserve(budget);
  std::vector<wfn_t<N>> members;
  members.reserve(group.size());
  for(const auto& [rep, score] : orbits) {
    // Materialize the orbit once from its representative
    members.clear();
    for(const auto& g : group) members.push_back(permute_orbitals(rep, g));
    std::sort(members.begin(), members.end(), bitset_less_comparator<N>{});
    members.erase(std::unique(members.begin(), members.end()), members.end());

    if(selected.size() + members.size() > budget) break;
    selected.insert(selected.end(), members.begin(), members.end());
  }

  std::sort(selected.begin(), selected.end(), bitset_less_comparator<N>{});
  return selected;
}

}  // namespace macis
