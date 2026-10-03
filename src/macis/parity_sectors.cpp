/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <cmath>
#include <iomanip>
#include <macis/parity_sectors.hpp>
#include <numeric>
#include <sstream>

namespace macis {

namespace {

struct UnionFind {
  std::vector<size_t> parent;
  explicit UnionFind(size_t n) : parent(n) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  size_t find(size_t x) {
    while(parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
  }
  void unite(size_t a, size_t b) { parent[find(a)] = find(b); }
};

std::string orb_list(const std::vector<uint32_t>& v) {
  std::ostringstream os;
  os << "[";
  for(size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
  os << "]";
  return os.str();
}

}  // namespace

ParityLabels build_parity_labels(size_t norb, size_t n_imp, size_t nbands,
                                 std::vector<double>& T,
                                 std::vector<double>* Td,
                                 std::vector<double>& V, double tol,
                                 std::ostream& log) {
  const size_t n = norb, n2 = n * n, n3 = n2 * n;
  if(nbands == 0 or n_imp == 0 or n_imp % nbands != 0 or n_imp > n)
    throw std::runtime_error(
        "ASCI.PARITY_SOLVE: CI.NIMP = " + std::to_string(n_imp) +
        " must be a nonzero multiple of CI.NBANDS = " + std::to_string(nbands) +
        " and at most the number of orbitals");
  if(nbands > 31)
    throw std::runtime_error("ASCI.PARITY_SOLVE: at most 31 bands");
  if(T.size() != n2 or V.size() != n2 * n2 or (Td and Td->size() != n2))
    throw std::runtime_error(
        "build_parity_labels: integral sizes do not match "
        "norb = " +
        std::to_string(n));
  const size_t nsites = n_imp / nbands;

  // ---- Groups: connected components of the one-body graph ----------------
  UnionFind uf(n);
  for(size_t q = 0; q < n; ++q)
    for(size_t p = 0; p < n; ++p)
      if(p != q and (std::abs(T[p + q * n]) > tol or
                     (Td and std::abs((*Td)[p + q * n]) > tol)))
        uf.unite(p, q);
  for(size_t i = 0; i < n_imp; ++i) uf.unite(i, (i / nsites) * nsites);

  ParityLabels L;
  L.tol = tol;
  L.ngroups = nbands;
  L.group_orbs.resize(nbands);
  L.group_of.assign(n, -1);
  std::vector<int> band_of_root(n, -1);
  for(size_t b = 0; b < nbands; ++b) {
    const size_t r = uf.find(b * nsites);
    if(band_of_root[r] >= 0) {
      // Two bands in one component: find a bath orbital hybridized with both,
      // or else report the component, to make the message actionable
      std::ostringstream os;
      os << "ASCI.PARITY_SOLVE: bands " << band_of_root[r] << " and " << b
         << " are connected by one-body couplings larger than PARITY_TOL = "
         << std::scientific << std::setprecision(2) << tol
         << ", so the bath is not band-diagonal and band parity is not "
            "conserved.";
      for(size_t j = n_imp; j < n; ++j) {
        double c0 = 0., c1 = 0.;
        for(size_t i = 0; i < n_imp; ++i) {
          const double t = std::abs(T[i + j * n]);
          if(int(i / nsites) == band_of_root[r]) c0 = std::max(c0, t);
          if(i / nsites == b) c1 = std::max(c1, t);
        }
        if(c0 > tol and c1 > tol) {
          os << " Bath orbital " << j << " couples to band " << band_of_root[r]
             << " (|T| = " << c0 << ") and to band " << b << " (|T| = " << c1
             << ").";
          break;
        }
      }
      throw std::runtime_error(os.str());
    }
    band_of_root[r] = int(b);
  }
  std::vector<uint32_t> decoupled;
  for(size_t q = 0; q < n; ++q) {
    const int b = band_of_root[uf.find(q)];
    L.group_of[q] = b;
    if(b >= 0)
      L.group_orbs[b].push_back(uint32_t(q));
    else
      decoupled.push_back(uint32_t(q));
  }

  // Membership used by the checks: groups 0..nbands-1, then one pseudo-group
  // per decoupled orbital (its own count must be conserved too)
  std::vector<int> member(n);
  {
    int next = int(nbands);
    for(size_t q = 0; q < n; ++q)
      member[q] = L.group_of[q] >= 0 ? L.group_of[q] : next++;
  }

  // ---- One-body: clean the cross-group elements --------------------------
  auto clean_1body = [&](std::vector<double>& A) {
    for(size_t q = 0; q < n; ++q)
      for(size_t p = 0; p < n; ++p) {
        double& x = A[p + q * n];
        if(member[p] == member[q] or x == 0.) continue;
        // Larger ones would have merged the components above
        L.max_discarded = std::max(L.max_discarded, std::abs(x));
        x = 0.;
      }
  };
  clean_1body(T);
  if(Td) clean_1body(*Td);

  // ---- Two-body: parity check, clean, count check ------------------------
  bool counts = true;
  double worst = 0.;
  size_t worst_idx = 0;
  for(size_t s = 0; s < n; ++s)
    for(size_t r = 0; r < n; ++r)
      for(size_t q = 0; q < n; ++q)
        for(size_t p = 0; p < n; ++p) {
          double& x = V[p + q * n + r * n2 + s * n3];
          if(x == 0.) continue;
          const int mp = member[p], mq = member[q], mr = member[r],
                    ms = member[s];
          // Parity: each group must appear an even number of times among the
          // four indices. Pair the indices up by group.
          bool even;
          if(mp == mq)
            even = (mr == ms);
          else if(mp == mr)
            even = (mq == ms);
          else if(mp == ms)
            even = (mq == mr);
          else
            even = false;
          if(!even) {
            if(std::abs(x) > tol) {
              if(std::abs(x) > worst)
                worst = std::abs(x), worst_idx = p + q * n + r * n2 + s * n3;
              continue;
            }
            L.max_discarded = std::max(L.max_discarded, std::abs(x));
            x = 0.;
            continue;
          }
          // Counts: (pq|rs) = a+_p a+_r a_s a_q moves an electron between
          // groups unless the creators and annihilators match group-wise
          if(std::abs(x) > tol and
             !((mp == mq and mr == ms) or (mp == ms and mr == mq)))
            counts = false;
        }
  if(worst > 0.) {
    const size_t p = worst_idx % n, q = (worst_idx / n) % n,
                 r = (worst_idx / n2) % n, s = worst_idx / n3;
    std::ostringstream os;
    os << "ASCI.PARITY_SOLVE: the two-body integral (" << p << " " << q << "|"
       << r << " " << s << ") = " << std::scientific << std::setprecision(3)
       << V[worst_idx]
       << " changes a band parity and exceeds PARITY_TOL = " << tol
       << ". Band parity is not conserved by this Hamiltonian.";
    throw std::runtime_error(os.str());
  }
  L.counts_conserved = counts;

  // ---- Report ------------------------------------------------------------
  log << "PARITY_LABELS groups = [";
  for(size_t b = 0; b < nbands; ++b)
    log << (b ? "," : "") << orb_list(L.group_orbs[b]);
  log << "] decoupled = " << orb_list(decoupled)
      << " max_discarded = " << std::scientific << std::setprecision(1)
      << L.max_discarded << std::defaultfloat
      << " counts_conserved = " << (counts ? "T" : "F") << std::endl;
  if(!decoupled.empty())
    log << "WARNING: PARITY_SOLVE: orbital(s) " << orb_list(decoupled)
        << " are decoupled from every band. Their occupation is conserved on "
           "its own and is taken from the energy-ordered seed; it is not "
           "searched over."
        << std::endl;
  if(counts)
    log << "WARNING: PARITY_SOLVE: no interaction term moves electrons "
           "between bands (no pair hopping), so the band counts are "
           "conserved, which is finer than parity. A count-sector trap is "
           "still possible and is not handled."
        << std::endl;
  return L;
}

}  // namespace macis
