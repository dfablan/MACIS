/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

/**
 * @brief Collection of routines to compute dynamical properties of a
 *        correlated wave function expressed as a list of determinants.
 *        Provides the resolvent expectation value evaluated directly on a
 *        reference (e.g. ground) state, and on static operators applied to
 *        that state (currently Sz on the impurity orbitals).
 *
 * @date 29/06/2026
 */
#pragma once
#include <Eigen/Core>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "macis/csr_hamiltonian.hpp"
#include "macis/gf/gf.hpp"       // GF_Diag, GFSettings
#include "macis/gf/lanczos.hpp"  // SparsexDistSpMatOp
#include "macis/hamiltonian_generator.hpp"
#include "macis/observables/impurity_rdm.hpp"  // decompose_det

namespace macis {

/**
 * @brief Computes the retarded resolvent expectation value of the Hamiltonian
 *        directly on a reference wave function, in its own (N-particle)
 *        determinant basis:
 *
 *          R(w) = <wfn0| 1 / (w - (H - E0)) |wfn0>
 *
 *        Unlike RunGFCalc, this does NOT build an N+/-1 determinant subspace.
 *        The Hamiltonian is built over the same determinant basis in which
 *        wfn0 is expressed (base_dets), and the resolvent is evaluated on
 *        wfn0 itself via the single-band Lanczos continued fraction (GF_Diag,
 *        called with ispart = true to select the retarded branch).
 *
 *        Note: the |wfn0|^2 numerator is handled internally by the Lanczos
 *        routine (betas[0] = ||wfn0||), so wfn0 need not be normalized.
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 * @tparam index_t: Integer index type for the sparse Hamiltonian.
 *
 * @param[in] const Eigen::VectorXd &wfn0: Reference wave function, expressed
 *            in the base_dets determinant basis.
 * @param[in] HamiltonianGenerator<nbits> &Hgen: Generator of Hamiltonian
 *            matrix elements.
 * @param[in] const std::vector<std::bitset<nbits>> &base_dets: Determinant
 *            basis describing wfn0.
 * @param[in] double E0: Reference state energy, used to shift the resolvent.
 * @param[in] const std::vector<std::complex<double>> &ws: Frequency grid over
 *            which to evaluate the resolvent.
 * @param[in] const GFSettings &settings: Parameters. Uses nLanIts (max Lanczos
 *            iterations) and saveGFmats (dump Lanczos alpha/beta coefficients).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 *
 * @date 29/06/2026
 */
template <size_t nbits, typename index_t = int32_t>
std::vector<std::complex<double>> RunResolventGS(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets, double E0,
    const std::vector<std::complex<double>> &ws, const GFSettings &settings) {
  const double h_el_tol = 1.E-6;
  int nLanIts = settings.nLanIts;
  if(int(base_dets.size()) < nLanIts) nLanIts = base_dets.size();

  // Build the Hamiltonian over the reference determinant basis (NOT a gf_dets
  // subspace) and wrap it as a MatOp for the Lanczos routine. A mutable copy of
  // base_dets is needed because make_dist_csr_hamiltonian takes non-const
  // wavefunction iterators.
  std::vector<std::bitset<nbits>> dets(base_dets);
  auto hamil = make_dist_csr_hamiltonian<index_t>(MPI_COMM_WORLD, dets.begin(),
                                                  dets.end(), Hgen, h_el_tol);
  SparsexDistSpMatOp hamil_wrap(hamil);

  // Single-vector continued-fraction resolvent on wfn0 itself.
  // ispart = true -> retarded branch 1/(w - (H - E0)).
  std::vector<std::complex<double>> R;
  GF_Diag<SparsexDistSpMatOp>(wfn0, hamil_wrap, ws, R, E0, /*ispart=*/true,
                              nLanIts, settings.saveGFmats, "resolvent_");
  return R;
}

/**
 * @brief Applies a diagonal (in the determinant basis) operator O to a wave
 *        function by rescaling each determinant coefficient. The result is a
 *        new coefficient vector v with v[k] = wfn0[k] * scalar_fn(dets[k]),
 *        i.e. v = O |wfn0> when O |D_k> = scalar_fn(D_k) |D_k>.
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 * @tparam ScalarFn: Callable (const std::bitset<nbits>&) -> double returning
 *         the diagonal eigenvalue of O on a given determinant.
 *
 * @param[in] const Eigen::VectorXd &wfn0: Input coefficient vector.
 * @param[in] const std::vector<std::bitset<nbits>> &dets: Determinant basis,
 *            with dets[k] the determinant of coefficient wfn0[k].
 * @param[in] ScalarFn scalar_fn: Per-determinant diagonal eigenvalue of O.
 *
 * @returns Eigen::VectorXd: The rescaled coefficient vector O |wfn0>.
 *
 * @date 29/06/2026
 */
template <size_t nbits, class ScalarFn>
Eigen::VectorXd apply_diagonal_operator(
    const Eigen::VectorXd &wfn0, const std::vector<std::bitset<nbits>> &dets,
    ScalarFn scalar_fn) {
  assert(wfn0.size() == Eigen::Index(dets.size()));
  Eigen::VectorXd out(wfn0.size());
  for(Eigen::Index k = 0; k < wfn0.size(); ++k)
    out[k] = wfn0[k] * scalar_fn(dets[k]);
  return out;
}

/**
 * @brief Which combination of spin-up / spin-down occupations a diagonal
 *        impurity operator accumulates on each orbital (see
 *        weighted_imp_value):
 *
 *          Charge : O = sum_i w_i ( n_{i,up} + n_{i,dn} )
 *          Spin   : O = sum_i w_i ( n_{i,up} - n_{i,dn} ) / 2
 *
 * @date 18/09/2026
 */
enum class DiagChannel { Charge, Spin };

/**
 * @brief Diagonal eigenvalue of a per-orbital-weighted impurity operator on a
 *        determinant:
 *
 *          Charge : O |D> = ( sum_i w_i (n_{i,up} + n_{i,dn}) ) |D>
 *          Spin   : O |D> = ( sum_i w_i (n_{i,up} - n_{i,dn}) / 2 ) |D>
 *
 *        where the sum runs over impurity orbitals i = 0 .. n_imp-1 and
 *        n_{i,up}/n_{i,dn} are the occupations of impurity orbital i in D.
 *        Reuses decompose_det (macis/observables/impurity_rdm.hpp) as the
 *        authoritative definition of the impurity orbitals.
 *
 *        decompose_det packs impurity orbital occupations in REVERSE: orbital
 *        i lives at bit (n_imp - 1 - i) of the packed imp_up/imp_dn words
 *        (decompose_det, impurity_rdm.hpp:42-45). This function un-reverses
 *        that mapping so that w[i] is the weight of orbital i as laid out by
 *        the caller (band-major, site-minor; see make_orbital_cartan_weights
 *        / make_staggered_spin_weights), not of whatever bit position
 *        decompose_det happens to store it at.
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 *
 * @param[in] const std::bitset<nbits> &det: Determinant.
 * @param[in] const std::vector<double> &w: Per-impurity-orbital weight, size
 *            n_imp.
 * @param[in] DiagChannel ch: Charge or Spin channel (see above).
 * @param[in] size_t n_imp: Number of impurity orbitals.
 * @param[in] size_t n_active: Number of active orbitals.
 *
 * @returns double: The diagonal eigenvalue of O on det.
 *
 * @date 18/09/2026
 */
template <size_t nbits>
inline double weighted_imp_value(const std::bitset<nbits> &det,
                                 const std::vector<double> &w, DiagChannel ch,
                                 size_t n_imp, size_t n_active) {
  assert(w.size() == n_imp);
  auto d = macis::decompose_det<nbits>(det, n_imp, n_active);
  double val = 0.0;
  for(size_t i = 0; i < n_imp; ++i) {
    // Undo decompose_det's reversed bit packing: orbital i is at bit
    // n_imp - 1 - i of imp_up/imp_dn.
    const size_t bit = n_imp - 1 - i;
    const double n_up = double((d.imp_up >> bit) & 1ULL);
    const double n_dn = double((d.imp_dn >> bit) & 1ULL);
    if(ch == DiagChannel::Charge)
      val += w[i] * (n_up + n_dn);
    else
      val += w[i] * 0.5 * (n_up - n_dn);
  }
  return val;
}

/**
 * @brief Diagonal eigenvalue of the impurity Sz operator on a determinant:
 *
 *          Sz_imp |D> = 0.5 * (n_up_imp - n_dn_imp) |D>
 *
 *        where n_up_imp / n_dn_imp are the numbers of spin-up / spin-down
 *        electrons occupying the impurity orbitals in D. Special case of
 *        weighted_imp_value with unit weight on every impurity orbital and
 *        ch = DiagChannel::Spin (equivalently, make_uniform_spin_weights).
 *
 *        Sz_imp is invariant under the block-diagonal, spin-conserving
 *        natural-orbital rotations used here: such rotations only redistribute
 *        occupation among the impurity orbitals, leaving the per-spin impurity
 *        electron counts (and hence this value) unchanged.
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 *
 * @param[in] const std::bitset<nbits> &det: Determinant.
 * @param[in] size_t n_imp: Number of impurity orbitals.
 * @param[in] size_t n_active: Number of active orbitals.
 *
 * @returns double: 0.5 * (n_up_imp - n_dn_imp) for det.
 *
 * @date 29/06/2026
 */
template <size_t nbits>
inline double sz_imp_value(const std::bitset<nbits> &det, size_t n_imp,
                           size_t n_active) {
  const std::vector<double> w(n_imp, 1.0);
  return weighted_imp_value<nbits>(det, w, DiagChannel::Spin, n_imp,
                                   n_active);
}

/**
 * @brief Uniform (q = 0) spin weight vector: w_i = 1 for every impurity
 *        orbital i, over all bands and sites. Passed with
 *        DiagChannel::Spin, this reproduces sz_imp_value / the existing
 *        Sz_imp resolvent, i.e. Sz(site 0) + Sz(site 1) + ... for a
 *        multi-site cluster.
 *
 * @param[in] size_t nbands: Number of impurity bands.
 * @param[in] size_t nsites: Number of impurity sites.
 *
 * @returns std::vector<double>: Weight vector of length nbands * nsites.
 *
 * @date 18/09/2026
 */
inline std::vector<double> make_uniform_spin_weights(size_t nbands,
                                                      size_t nsites) {
  return std::vector<double>(nbands * nsites, 1.0);
}

/**
 * @brief Staggered (q = pi) spin weight vector for a two-site impurity
 *        cluster: +1 on site 0, -1 on site 1, uniformly over bands. Passed
 *        with DiagChannel::Spin, this probes Sz(site 0) - Sz(site 1), the
 *        inter-site antiferromagnetic / singlet channel that the uniform
 *        (make_uniform_spin_weights) resolvent cannot see (see §7.4 of
 *        PLAN_orbital_resolvent.md).
 *
 * @param[in] size_t nbands: Number of impurity bands.
 * @param[in] size_t nsites: Number of impurity sites. Must be 2.
 *
 * @returns std::vector<double>: Weight vector of length nbands * nsites.
 *
 * @throws std::runtime_error if nsites != 2.
 *
 * @date 18/09/2026
 */
inline std::vector<double> make_staggered_spin_weights(size_t nbands,
                                                        size_t nsites) {
  if(nsites != 2)
    throw std::runtime_error(
        "make_staggered_spin_weights: requires nsites == 2, got nsites = " +
        std::to_string(nsites));
  const size_t n_imp = nbands * nsites;
  std::vector<double> w(n_imp);
  // Orbital index layout is band-major, site-minor: i = site + nsites*band
  // (see PLAN_orbital_resolvent.md §2.1).
  for(size_t i = 0; i < n_imp; ++i) w[i] = (i % nsites == 0) ? 1.0 : -1.0;
  return w;
}

/**
 * @brief Orbital (band) isospin Cartan generator weight vector, i.e. the
 *        diagonal SU(3) generators T^3 / T^8 restricted to the CHARGE
 *        channel:
 *
 *          T^3 = (1/2)           (n_1 - n_2)
 *          T^8 = (1/(2*sqrt(3))) (n_1 + n_2 - 2*n_3)
 *
 *        applied uniformly across sites. Both are traceless
 *        (sum_i w_i == 0), which is what isolates orbital *polarization*
 *        from the impurity charge channel (see §1.4 of
 *        PLAN_orbital_resolvent.md). Pass the result with
 *        DiagChannel::Charge.
 *
 * @param[in] size_t nbands: Number of impurity bands. which = 3 requires
 *            nbands >= 2; which = 8 requires nbands == 3.
 * @param[in] size_t nsites: Number of impurity sites.
 * @param[in] int which: 3 for T^3, 8 for T^8.
 *
 * @returns std::vector<double>: Weight vector of length nbands * nsites.
 *
 * @throws std::runtime_error if nbands == 1 (no traceless orbital generator
 *         exists in a one-dimensional orbital space), if which is not 3 or
 *         8, or if which == 8 and nbands != 3.
 *
 * @date 18/09/2026
 */
inline std::vector<double> make_orbital_cartan_weights(size_t nbands,
                                                        size_t nsites,
                                                        int which) {
  if(nbands == 1)
    throw std::runtime_error(
        "make_orbital_cartan_weights: no traceless orbital generator exists "
        "for nbands == 1 (there is no orbital degree of freedom to "
        "screen)");
  if(which != 3 && which != 8)
    throw std::runtime_error(
        "make_orbital_cartan_weights: which must be 3 (T^3) or 8 (T^8), "
        "got which = " +
        std::to_string(which));
  if(which == 8 && nbands != 3)
    throw std::runtime_error(
        "make_orbital_cartan_weights: T^8 (which == 8) requires nbands == "
        "3, got nbands = " +
        std::to_string(nbands));

  // Per-band weight; bands beyond those named in the generator get weight 0.
  std::vector<double> band_w(nbands, 0.0);
  if(which == 3) {
    band_w[0] = 0.5;
    band_w[1] = -0.5;
  } else {  // which == 8, nbands == 3
    const double c = 1.0 / (2.0 * std::sqrt(3.0));
    band_w[0] = c;
    band_w[1] = c;
    band_w[2] = -2.0 * c;
  }

  const size_t n_imp = nbands * nsites;
  std::vector<double> w(n_imp);
  // Orbital index layout is band-major, site-minor: i = site + nsites*band,
  // so band = i / nsites (see PLAN_orbital_resolvent.md §2.1).
  for(size_t i = 0; i < n_imp; ++i) w[i] = band_w[i / nsites];

  const double sum = std::accumulate(w.begin(), w.end(), 0.0);
  assert(std::abs(sum) < 1e-10 &&
        "make_orbital_cartan_weights: generator is not traceless");
  (void)sum;
  return w;
}

/**
 * @brief Computes the retarded resolvent of the Hamiltonian on a diagonal
 *        (in the determinant basis) operator O applied to a reference (e.g.
 *        ground) state:
 *
 *          R(w) = <wfn0| O  1 / (w - (H - E0))  O |wfn0>
 *
 *        O is applied first via apply_diagonal_operator (rescaling each
 *        determinant coefficient by scalar_fn), and the single-vector
 *        resolvent of the resulting state v = O |wfn0> is then evaluated via
 *        RunResolventGS.
 *
 *        Note: v = O |wfn0> is generally NOT an eigenstate of H, and need not
 *        be normalized (the |v|^2 numerator is handled by the underlying
 *        Lanczos routine).
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 * @tparam index_t: Integer index type for the sparse Hamiltonian.
 * @tparam ScalarFn: Callable (const std::bitset<nbits>&) -> double returning
 *         the diagonal eigenvalue of O on a given determinant.
 *
 * @param[in] const Eigen::VectorXd &wfn0: Reference wave function, expressed
 *            in the base_dets determinant basis.
 * @param[in] HamiltonianGenerator<nbits> &Hgen: Generator of Hamiltonian
 *            matrix elements.
 * @param[in] const std::vector<std::bitset<nbits>> &base_dets: Determinant
 *            basis describing wfn0.
 * @param[in] ScalarFn scalar_fn: Per-determinant diagonal eigenvalue of O.
 * @param[in] size_t n_imp: Number of impurity orbitals used by scalar_fn
 *            (via decompose_det); only used for the packing guard below.
 * @param[in] double E0: Reference state energy, used to shift the resolvent.
 * @param[in] const std::vector<std::complex<double>> &ws: Frequency grid over
 *            which to evaluate the resolvent.
 * @param[in] const GFSettings &settings: Parameters (nLanIts, saveGFmats).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 *
 * @date 18/09/2026
 */
template <size_t nbits, typename index_t = int32_t, class ScalarFn>
std::vector<std::complex<double>> RunResolventDiagonal(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets, ScalarFn scalar_fn,
    size_t n_imp, double E0, const std::vector<std::complex<double>> &ws,
    const GFSettings &settings) {
  // decompose_det packs the impurity occupation into a uint64_t (n_imp bits
  // per spin), so 2 * n_imp must fit in 64 bits.
  if(2 * n_imp > 64)
    throw std::runtime_error(
        "RunResolventDiagonal: 2*n_imp > 64 not supported");

  // Build v = O |wfn0> by rescaling each determinant coefficient.
  Eigen::VectorXd v =
      apply_diagonal_operator<nbits>(wfn0, base_dets, scalar_fn);

  // If O |wfn0> vanishes (e.g. Sz_imp on an Sz_tot = 0 state with
  // n_imp == n_active, or any traceless operator on an orbitally
  // unpolarized determinant set), the resolvent is identically zero. Return
  // early to avoid feeding a zero start vector into the Lanczos routine
  // (which would divide by ||v||).
  const double zero_thresh = 1.E-12;
  if(v.squaredNorm() <= zero_thresh)
    return std::vector<std::complex<double>>(ws.size(),
                                             std::complex<double>(0., 0.));

  // Resolvent of the resulting state in the same determinant basis.
  return RunResolventGS<nbits, index_t>(v, Hgen, base_dets, E0, ws, settings);
}

/**
 * @brief Computes the retarded resolvent of the Hamiltonian on the impurity
 *        Sz operator applied to a reference (e.g. ground) state:
 *
 *          R(w) = <wfn0| Sz_imp  1 / (w - (H - E0))  Sz_imp |wfn0>
 *
 *        Thin wrapper over RunResolventDiagonal with scalar_fn = sz_imp_value.
 *        With E0 the ground-state energy, R(w) is the dynamical impurity
 *        Sz-Sz spin response, with poles at the spin excitation energies
 *        relative to the ground state.
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 * @tparam index_t: Integer index type for the sparse Hamiltonian.
 *
 * @param[in] const Eigen::VectorXd &wfn0: Reference wave function, expressed
 *            in the base_dets determinant basis.
 * @param[in] HamiltonianGenerator<nbits> &Hgen: Generator of Hamiltonian
 *            matrix elements.
 * @param[in] const std::vector<std::bitset<nbits>> &base_dets: Determinant
 *            basis describing wfn0.
 * @param[in] size_t n_imp: Number of impurity orbitals.
 * @param[in] size_t n_active: Number of active orbitals.
 * @param[in] double E0: Reference state energy (ground-state energy), used to
 *            shift the resolvent.
 * @param[in] const std::vector<std::complex<double>> &ws: Frequency grid over
 *            which to evaluate the resolvent.
 * @param[in] const GFSettings &settings: Parameters (nLanIts, saveGFmats).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 *
 * @date 29/06/2026
 */
template <size_t nbits, typename index_t = int32_t>
std::vector<std::complex<double>> RunResolventSz(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets, size_t n_imp,
    size_t n_active, double E0, const std::vector<std::complex<double>> &ws,
    const GFSettings &settings) {
  const auto sz_operator = [&](const std::bitset<nbits> &d) {
    return sz_imp_value<nbits>(d, n_imp, n_active);
  };
  return RunResolventDiagonal<nbits, index_t>(
      wfn0, Hgen, base_dets, sz_operator, n_imp, E0, ws, settings);
}

/**
 * @brief Computes the retarded resolvent of the Hamiltonian on a general
 *        per-orbital-weighted diagonal impurity operator O applied to a
 *        reference (e.g. ground) state:
 *
 *          R(w) = <wfn0| O  1 / (w - (H - E0))  O |wfn0>,
 *          O = weighted_imp_value(., w, ch, n_imp, n_active)
 *
 *        Thin wrapper over RunResolventDiagonal with
 *        scalar_fn = weighted_imp_value. Covers the orbital Cartan
 *        generators (make_orbital_cartan_weights, DiagChannel::Charge),
 *        staggered spin (make_staggered_spin_weights, DiagChannel::Spin),
 *        and, via make_uniform_spin_weights, the existing Sz_imp response
 *        (RunResolventSz is kept as a separate, unchanged-signature
 *        entry point for that case).
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 * @tparam index_t: Integer index type for the sparse Hamiltonian.
 *
 * @param[in] const Eigen::VectorXd &wfn0: Reference wave function, expressed
 *            in the base_dets determinant basis.
 * @param[in] HamiltonianGenerator<nbits> &Hgen: Generator of Hamiltonian
 *            matrix elements.
 * @param[in] const std::vector<std::bitset<nbits>> &base_dets: Determinant
 *            basis describing wfn0.
 * @param[in] const std::vector<double> &w: Per-impurity-orbital weight, size
 *            n_imp (see make_orbital_cartan_weights /
 *            make_staggered_spin_weights / make_uniform_spin_weights).
 * @param[in] DiagChannel ch: Charge or Spin channel.
 * @param[in] size_t n_imp: Number of impurity orbitals.
 * @param[in] size_t n_active: Number of active orbitals.
 * @param[in] double E0: Reference state energy (ground-state energy), used to
 *            shift the resolvent.
 * @param[in] const std::vector<std::complex<double>> &ws: Frequency grid over
 *            which to evaluate the resolvent.
 * @param[in] const GFSettings &settings: Parameters (nLanIts, saveGFmats).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 *
 * @date 18/09/2026
 */
template <size_t nbits, typename index_t = int32_t>
std::vector<std::complex<double>> RunResolventWeighted(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets,
    const std::vector<double> &w, DiagChannel ch, size_t n_imp,
    size_t n_active, double E0, const std::vector<std::complex<double>> &ws,
    const GFSettings &settings) {
  const auto op = [&](const std::bitset<nbits> &d) {
    return weighted_imp_value<nbits>(d, w, ch, n_imp, n_active);
  };
  return RunResolventDiagonal<nbits, index_t>(wfn0, Hgen, base_dets, op,
                                              n_imp, E0, ws, settings);
}

}  // namespace macis
