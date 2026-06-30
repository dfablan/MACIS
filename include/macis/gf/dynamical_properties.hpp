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
#include <complex>
#include <stdexcept>
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
  auto hamil = make_dist_csr_hamiltonian<index_t>(
      MPI_COMM_WORLD, dets.begin(), dets.end(), Hgen, h_el_tol);
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
    const Eigen::VectorXd &wfn0,
    const std::vector<std::bitset<nbits>> &dets, ScalarFn scalar_fn) {
  assert(wfn0.size() == Eigen::Index(dets.size()));
  Eigen::VectorXd out(wfn0.size());
  for(Eigen::Index k = 0; k < wfn0.size(); ++k)
    out[k] = wfn0[k] * scalar_fn(dets[k]);
  return out;
}

/**
 * @brief Diagonal eigenvalue of the impurity Sz operator on a determinant:
 *
 *          Sz_imp |D> = 0.5 * (n_up_imp - n_dn_imp) |D>
 *
 *        where n_up_imp / n_dn_imp are the numbers of spin-up / spin-down
 *        electrons occupying the impurity orbitals in D. Reuses decompose_det
 *        (macis/observables/impurity_rdm.hpp) as the authoritative definition
 *        of the impurity orbitals.
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
  auto d = macis::decompose_det<nbits>(det, n_imp, n_active);
  return 0.5 * (double(d.n_imp_up) - double(d.n_imp_dn));
}

/**
 * @brief Computes the retarded resolvent of the Hamiltonian on the impurity
 *        Sz operator applied to a reference (e.g. ground) state:
 *
 *          R(w) = <wfn0| Sz_imp  1 / (w - (H - E0))  Sz_imp |wfn0>
 *
 *        The operator Sz_imp is applied first (rescaling each determinant
 *        coefficient, see sz_imp_value), and the single-vector resolvent of
 *        the resulting state v = Sz_imp |wfn0> is then evaluated via
 *        RunResolventGS. With E0 the ground-state energy, R(w) is the
 *        dynamical impurity Sz-Sz spin response, with poles at the spin
 *        excitation energies relative to the ground state.
 *
 *        Note: v = Sz_imp |wfn0> is generally NOT an eigenstate of H, and need
 *        not be normalized (the |v|^2 numerator is handled by the underlying
 *        Lanczos routine).
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
  // decompose_det packs the impurity occupation into a uint64_t (n_imp bits
  // per spin), so 2 * n_imp must fit in 64 bits.
  if(2 * n_imp > 64)
    throw std::runtime_error("RunResolventSz: 2*n_imp > 64 not supported");

  // Build v = Sz_imp |wfn0> by rescaling each determinant coefficient.
  Eigen::VectorXd v = apply_diagonal_operator<nbits>(
      wfn0, base_dets, [&](const std::bitset<nbits> &d) {
        return sz_imp_value<nbits>(d, n_imp, n_active);
      });

  // If Sz_imp |wfn0> vanishes (e.g. n_imp == n_active and wfn0 has total
  // Sz = 0), the resolvent is identically zero. Return early to avoid feeding a
  // zero start vector into the Lanczos routine (which would divide by ||v||).
  const double zero_thresh = 1.E-12;
  if(v.squaredNorm() <= zero_thresh)
    return std::vector<std::complex<double>>(ws.size(),
                                             std::complex<double>(0., 0.));

  // Resolvent of the resulting state in the same determinant basis.
  return RunResolventGS<nbits, index_t>(v, Hgen, base_dets, E0, ws, settings);
}

}  // namespace macis
