/**
 * @file charge_sectors.hpp
 * @brief Charge-sector exploration, and a chemical-potential search that also
 * finds the ground-state charge sector.
 *
 * The FCIDUMP carries -mu on the impurity diagonal, so E(CI) of a sector
 * (NALPHA, NBETA) is the grand potential Omega(N) at that mu. At a fixed mu
 * the sectors therefore compare directly and the lowest one is the ground
 * state. This header holds the pieces shared by the `explore_charge_sectors`
 * driver (scan at a fixed mu) and Fix_Mu_sectors (mu search + scan):
 *
 *  - seeding a sector from its solved neighbour by one c^dagger / c,
 *  - solving one sector (cold, or seeded and handed to the production ASCI
 *    path as a guess wavefunction), and
 *  - SectorScan, the walk / window search over N.
 *
 * Only the minimal-|S_z| sector of each N is solved: with SU(2) it contains
 * every multiplet.
 */

#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "macis/impurity_solver.hpp"

namespace macis {

/// Settings of the sector search (DOP.SECTOR_* keys of run_asci_impsolv_dop).
struct ChargeSectorSettings {
  bool enabled = true;  ///< DOP.SECTOR_SEARCH
  bool warm = true;     ///< seed neighbours from the solved sector
  bool warm_nrots0 =
      false;  ///< explore_charge_sectors only: NROTS = 0 everywhere
  size_t margin =
      2;  ///< walk until the minimum has `margin` sectors on each side
  double etol = 1e-4;  ///< a sector must lie more than this below to replace
                       ///< the current one
  size_t seed_parents =
      0;  ///< parent determinants used for a seed (0: NCDETS_MAX)
  size_t seed_size = 0;   ///< determinants kept in a seed (0: NTDETS_MAX)
  size_t max_switch = 4;  ///< cap on sector switches in Fix_Mu_sectors
  std::string workdir =
      "charge_sectors";  ///< scratch dir for seed files and solver side files
  std::ostream* out = &std::cout;
};

namespace charge_sectors {

/// An orbital basis, as the cumulative rotation from the original active
/// orbitals (n_active x n_active, column-major, the convention of
/// impurity_params::orb_rot). Empty means the original orbitals.
using basis_t = std::vector<double>;

/// Minimal-|S_z| split of N electrons. beta_heavy picks (k, k+1) instead of
/// (k+1, k) for odd N; it matters only for spin-dependent integrals.
inline size_t split_alpha(size_t n, bool beta_heavy = false) {
  return beta_heavy ? n / 2 : (n + 1) / 2;
}
inline size_t split_beta(size_t n, bool beta_heavy = false) {
  return beta_heavy ? (n + 1) / 2 : n / 2;
}

std::string sector_str(size_t a, size_t b);

/// Active integrals and settings every sector starts from.
template <size_t N>
struct Pristine {
  std::vector<double> T_active, V_active, Td_active;
  macis::ASCISettings asci_settings;
  bool just_singles;
};

/// Where a sector's starting wavefunction comes from.
template <size_t N>
struct SeedSource {
  enum Kind { Cold, Parent, File } kind = Cold;
  std::string label = "cold";
  const std::vector<wfn_t<N>>* dets =
      nullptr;  ///< Parent: the parent's wavefunction
  const std::vector<double>* C = nullptr;
  size_t pa = 0, pb = 0;
  std::string fname;  ///< File
  /// Orbital basis the seed determinants are written in (Parent, File)
  const basis_t* U = nullptr;
};

template <size_t N>
struct SectorResult {
  size_t na = 0, nb = 0;
  bool converged = false;
  std::string status, seed;
  double E = NAN, E_seed = NAN, n_band = NAN, time_s = 0;
  size_t ndets = 0;
  std::vector<wfn_t<N>>
      dets;  ///< kept while the sector may still seed a neighbour
  std::vector<double> C;
  basis_t U;  ///< orbital basis dets/C are written in
};

template <size_t N>
struct SectorContext {
  impurity_params<N>* p;
  Pristine<N> pristine;
  ChargeSectorSettings opt;
  size_t seed_parents, seed_size;
  size_t nrots_cold;  ///< NROTS of a solve that starts cold
  std::ostream* out;
  bool use_ed = false;  ///< solve sectors with CAS instead of ASCI (no seeding)
  bool beta_heavy = false;  ///< odd-N sectors are (k, k+1)
};

/// Active integrals rebuilt from p.T / p.V (unrotated, at the current mu).
template <size_t N>
void rebuild_active(impurity_params<N>& p);

/// Rotate the active integrals in place into the basis @p U (T <- U^T T U, same
/// for Td and all four indices of V) and turn singles-only off in p.
template <size_t N>
void rotate_active(impurity_params<N>& p, const basis_t& U);

/// Solve one sector; a seeded solve that fails (other than refinement not
/// converging) is retried cold.
template <size_t N>
SectorResult<N> solve_sector(SectorContext<N>& ctx, size_t na, size_t nb,
                             const SeedSource<N>& src);

/// Solved sectors, and the walk / window search over N.
template <size_t N>
class SectorScan {
 public:
  SectorScan(SectorContext<N>& ctx, size_t N0, size_t Nmax)
      : ctx_(ctx), N0_(N0), Nmax_(Nmax) {}

  std::map<size_t, SectorResult<N>> res;

  bool in_range(long n) const { return n >= 0 and n <= long(Nmax_); }

  /// Solve N from the solved neighbour @p Np (or cold).
  void solve(size_t n, std::optional<size_t> Np);
  void solve_reference(const SeedSource<N>& src);
  /// Take an already solved sector (the reference of a mu search).
  void set(size_t n, SectorResult<N> r);

  /// Lowest converged N.
  std::optional<size_t> argmin() const;
  size_t lo() const { return res.begin()->first; }
  size_t hi() const { return res.rbegin()->first; }

  /// Solve N0-1, N0+1, then extend on whichever side of the minimum has fewer
  /// than @p margin higher sectors, until the minimum is bracketed by
  /// @p margin sectors on each side or the range ends.
  void walk(long margin);
  /// Solve N0-W .. N0+W outward from N0.
  void window(long W);

 private:
  void prune();
  SectorContext<N>& ctx_;
  size_t N0_, Nmax_;
};

void write_header(std::ostream& os, const std::string& lead);
template <size_t N>
void write_row(std::ostream& os, size_t n, const SectorResult<N>& r,
               double Emin);

}  // namespace charge_sectors

/**
 * @brief Writes the ground-state sector to a small text file (rank 0 only):
 *   GROUND_SECTOR NALPHA = a NBETA = b N = n E = e MU = x
 * followed by @p extra (comment lines). @p p holds the accepted solution.
 */
template <size_t N>
void write_ground_sector_file(const std::string& fname,
                              const impurity_params<N>& p, double mu,
                              const std::string& extra);

/**
 * @brief Fixes mu (Fix_Mu_der / Fix_Mu_noder) AND finds the ground-state
 * sector.
 *
 * Outer loop around the existing mu search, in the sector (NALPHA, NBETA) of
 * @p params:
 *  1. mu search in the current sector N.
 *  2. At the converged mu, scan the neighbouring sectors (SectorScan::walk,
 *     `margin`). Every sector is solved with the same integrals, so the
 * energies compare directly.
 *  3. Accept N if no scanned sector lies more than `etol` below it. Otherwise
 *     switch to the lowest sector and go back to 1, starting from the same mu.
 *
 * Throws if a sector would be searched twice (the target filling lies in the
 * jump between two sectors, where no ground state has it) or after
 * `max_switch` switches.
 *
 * On return @p params holds the solution of the accepted sector at the returned
 * mu, as after Fix_Mu_der / Fix_Mu_noder, and GS_charge_sector.dat is written
 * to the working directory.
 */
template <size_t N>
double Fix_Mu_sectors(const std::string& method_name, bool deriv,
                      double& init_mu, impurity_params<N>* params,
                      const ChargeSectorSettings& settings);

}  // namespace macis
