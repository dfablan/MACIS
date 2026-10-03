/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstdint>
#include <macis/types.hpp>
#include <memory>
#include <string>
#include <vector>

namespace macis {

/**
 *  @brief Band-parity labels of the active orbitals.
 *
 *  For a band-diagonal bath every band -- its impurity orbitals plus the bath
 *  orbitals hybridized with them -- is an orbital group g, and H changes each
 *  group's electron count N_g only by even amounts (pair hopping), so the
 *  parity (-1)^{N_g} is an exact label of every determinant. Built once per
 *  Hamiltonian by build_parity_labels (macis/parity_sectors.hpp).
 */
struct ParityLabels {
  size_t ngroups = 0;
  // active orbital -> group; -1 = decoupled ("dark") orbital, in no group
  std::vector<int> group_of;
  // group -> its active orbitals
  std::vector<std::vector<uint32_t>> group_orbs;
  // largest integral that broke a parity and was zeroed by the cleaning
  double max_discarded = 0.;
  // no interaction term moves electrons between groups: the counts N_g are
  // conserved, which is finer than parity and not handled here
  bool counts_conserved = false;
};

/**
 *  @brief Determinant -> parity key: bit g is N_g mod 2, alpha and beta summed.
 *  Decoupled orbitals are in no mask and do not enter the key.
 */
template <size_t N>
struct ParityMasks {
  std::vector<wfn_t<N>> mask;  // per group: alpha bits | beta bits << N/2

  explicit ParityMasks(const ParityLabels& L) : mask(L.ngroups) {
    for(size_t g = 0; g < L.ngroups; ++g)
      for(auto q : L.group_orbs[g]) {
        mask[g].set(q);
        mask[g].set(q + N / 2);
      }
  }

  uint32_t key(const wfn_t<N>& d) const {
    uint32_t k = 0;
    for(size_t g = 0; g < mask.size(); ++g)
      k |= uint32_t((d & mask[g]).count() & 1u) << g;
    return k;
  }
};

/// Human-readable key, one letter per group: "(o,e)" = N_0 odd, N_1 even
inline std::string parity_key_string(uint32_t key, size_t ngroups) {
  std::string s = "(";
  for(size_t g = 0; g < ngroups; ++g) {
    if(g) s += ",";
    s += ((key >> g) & 1u) ? "o" : "e";
  }
  return s + ")";
}

/**
 *  @brief Sector constraint for one ASCI solve. Set only by the parity-sector
 *  wrapper; asci_search drops every candidate whose key differs from `key`.
 *  ASCISettings is not templated on N, so this holds orbital lists, and the
 *  search builds ParityMasks<N> from them.
 */
struct ParityTarget {
  std::shared_ptr<const ParityLabels> labels;
  uint32_t key = 0;
};

}  // namespace macis
