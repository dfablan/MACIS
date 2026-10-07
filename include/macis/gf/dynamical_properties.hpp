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
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "macis/csr_hamiltonian.hpp"
#include "macis/gf/gf.hpp"       // GF_Diag, GFSettings
#include "macis/gf/lanczos.hpp"  // SparsexDistSpMatOp
#include "macis/hamiltonian_generator.hpp"
#include "macis/observables/impurity_rdm.hpp"  // decompose_det
#include "macis/sd_operations.hpp"

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
 * @brief Which combination of spin-up / spin-down occupations an impurity
 *        operator accumulates on each orbital.
 *
 *        For the diagonal operators of weighted_imp_value:
 *
 *          Charge : O = sum_i w_i ( n_{i,up} + n_{i,dn} )
 *          Spin   : O = sum_i w_i ( n_{i,up} - n_{i,dn} ) / 2
 *
 *        For the orbital bilinears of apply_orbital_bilinear (no factor 1/2):
 *
 *          Charge : N_{mu nu} = c^+_{mu,up} c_{nu,up} + c^+_{mu,dn} c_{nu,dn}
 *          Spin   : S_{mu nu} = c^+_{mu,up} c_{nu,up} - c^+_{mu,dn} c_{nu,dn}
 *
 * @date 18/09/2026
 */
enum class DiagChannel { Charge, Spin };

/**
 * @brief Applies the orbital bilinear O_{mu nu} = sum_sigma s_sigma
 *        c^+_{mu,sigma} c_{nu,sigma} to a wave function, with s_up = +1 and
 *        s_dn = -1 in the Spin channel (S_{mu nu}) or s_dn = +1 in the Charge
 *        channel (N_{mu nu}). Images that fall outside `dets` are dropped, so
 *        the result is the projection of O_{mu nu}|wfn0> onto the basis. For
 *        mu = nu, S_{mu mu} = 2 S_z^mu and N_{mu mu} = n_mu.
 *
 * @tparam nbits: Number of bits in the Slater determinant bitset type.
 *
 * @param[in] const Eigen::VectorXd &wfn0: Input coefficient vector.
 * @param[in] const std::vector<std::bitset<nbits>> &dets: Determinant basis,
 *            with dets[k] the determinant of coefficient wfn0[k].
 * @param[in] det_index: Lookup from determinant to its position in dets.
 * @param[in] size_t mu, nu: Impurity orbitals of the creator / annihilator.
 * @param[in] DiagChannel ch: Spin (S_{mu nu}) or Charge (N_{mu nu}).
 *
 * @returns Eigen::VectorXd: The projected coefficient vector O_{mu nu}|wfn0>.
 */
template <size_t nbits>
Eigen::VectorXd apply_orbital_bilinear(
    const Eigen::VectorXd &wfn0, const std::vector<std::bitset<nbits>> &dets,
    const std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>>
        &det_index,
    size_t mu, size_t nu, DiagChannel ch) {
  assert(wfn0.size() == Eigen::Index(dets.size()));
  const double dn_sign = (ch == DiagChannel::Spin) ? -1.0 : 1.0;
  Eigen::VectorXd out = Eigen::VectorXd::Zero(wfn0.size());
  for(Eigen::Index k = 0; k < wfn0.size(); ++k) {
    for(size_t spin = 0; spin < 2; ++spin) {
      const size_t offset = spin ? nbits / 2 : 0;
      const size_t p = mu + offset;
      const size_t q = nu + offset;
      const auto &det = dets[k];
      if(!det.test(q) || (mu != nu && det.test(p))) continue;
      std::bitset<nbits> image(det);
      double sign = 1.0;
      if(mu != nu) {
        image.flip(q);
        image.flip(p);
        sign = single_excitation_sign(det, p, q);
      }
      const auto it = det_index.find(image);
      if(it != det_index.end())
        out[it->second] += (spin ? dn_sign : 1.0) * sign * wfn0[k];
    }
  }
  return out;
}

// Image of O_{mu nu}|wfn0> relative to a determinant basis: the captured
// fraction (see orbital_bilinear_captured_fraction) and the images that fall
// outside the basis ("leaked"), with their accumulated coefficients, in
// determinant order.
template <size_t nbits>
struct OrbitalBilinearImage {
  double capture = 1.0;
  std::vector<std::bitset<nbits>> leaked;
  std::vector<double> leaked_amplitude;
};

template <size_t nbits>
OrbitalBilinearImage<nbits> orbital_bilinear_image(
    const Eigen::VectorXd &wfn0, const std::vector<std::bitset<nbits>> &dets,
    const std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>>
        &det_index,
    size_t mu, size_t nu, DiagChannel ch) {
  assert(wfn0.size() == Eigen::Index(dets.size()));
  const double dn_sign = (ch == DiagChannel::Spin) ? -1.0 : 1.0;
  std::map<std::bitset<nbits>, double, bitset_less_comparator<nbits>> images;
  for(Eigen::Index k = 0; k < wfn0.size(); ++k) {
    for(size_t spin = 0; spin < 2; ++spin) {
      const size_t offset = spin ? nbits / 2 : 0;
      const size_t p = mu + offset;
      const size_t q = nu + offset;
      const auto &det = dets[k];
      if(!det.test(q) || (mu != nu && det.test(p))) continue;
      std::bitset<nbits> image(det);
      double sign = 1.0;
      if(mu != nu) {
        image.flip(q);
        image.flip(p);
        sign = single_excitation_sign(det, p, q);
      }
      images[image] += (spin ? dn_sign : 1.0) * sign * wfn0[k];
    }
  }
  OrbitalBilinearImage<nbits> out;
  double captured_norm = 0.0;
  double total_norm = 0.0;
  for(const auto &[image, coefficient] : images) {
    const double norm = coefficient * coefficient;
    total_norm += norm;
    if(det_index.find(image) != det_index.end()) {
      captured_norm += norm;
    } else {
      out.leaked.push_back(image);
      out.leaked_amplitude.push_back(coefficient);
    }
  }
  out.capture = total_norm > 0.0 ? captured_norm / total_norm : 1.0;
  return out;
}

// Estimate the ratio between the norm of the in-basis component of
// O_{mu nu}|wfn0> and the norm of the full O_{mu nu}|wfn0> vector, with
// O_{mu nu} the bilinear of apply_orbital_bilinear. Some determinants can be
// lost if the determinant basis is not complete, so this is a measure of how
// much of O_{mu nu}|wfn0> is captured by the basis. A value of 1.0 means all
// determinants are captured, while a value of 0.0 means none are captured.
// The channel matters: spin-up and spin-down hops can reach the same image,
// and their relative sign decides whether they interfere constructively.
template <size_t nbits>
double orbital_bilinear_captured_fraction(
    const Eigen::VectorXd &wfn0, const std::vector<std::bitset<nbits>> &dets,
    const std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>>
        &det_index,
    size_t mu, size_t nu, DiagChannel ch) {
  return orbital_bilinear_image(wfn0, dets, det_index, mu, nu, ch).capture;
}

// Adds the leaked images of one seed to the growth seeds of the basis
// expansion. A determinant leaked by several seeds keeps its largest
// |amplitude|: amplitudes of different operators are not summed, so they
// cannot cancel and drop a determinant from the growth.
template <size_t nbits>
void merge_leaked_images(
    std::map<std::bitset<nbits>, double, bitset_less_comparator<nbits>> &merged,
    const OrbitalBilinearImage<nbits> &image) {
  for(size_t k = 0; k < image.leaked.size(); ++k) {
    double &amp = merged[image.leaked[k]];
    amp = std::max(amp, std::abs(image.leaked_amplitude[k]));
  }
}

/**
 * @brief Active orbitals of the GF basis growth: those whose per-spin
 *        occupation lies in [asThres, 1 - asThres]. Same rule as
 *        get_GF_basis_AS_1El.
 */
inline std::vector<uint32_t> active_space_orbitals(
    const std::vector<double> &occs, double asThres) {
  std::vector<uint32_t> as_orbs;
  for(size_t i = 0; i < occs.size(); i++)
    if(occs[i] >= asThres && occs[i] <= (1. - asThres)) as_orbs.push_back(i);
  return as_orbs;
}

/**
 * @brief Grows a determinant set from `seeds` by layers of active-space single
 *        excitations, as get_GF_basis_AS_1El does for the GF basis:
 *
 *        - every seed not in `exclude` is kept;
 *        - layer 1 grows only from the seeds with |amplitude| >= GFseedThres,
 *          each later layer from the determinants the previous one added;
 *        - settings.tot_SD layers, stopping once more than
 *          settings.trunc_size determinants are kept (trunc_size = 0: no cap).
 *
 *        Determinants in `exclude` (the base basis) are never added or grown
 *        from. The result is in insertion order and contains no duplicates.
 *
 * @param[in] seeds, amplitudes: Seed determinants (no duplicates) and the
 *            coefficients that gate layer 1.
 * @param[in] exclude: Determinants already in the basis.
 * @param[in] as_orbs: Active orbitals (see active_space_orbitals).
 * @param[in] norbs: Number of spatial orbitals passed to
 *            generate_singles_spin_as.
 */
template <size_t nbits>
std::vector<std::bitset<nbits>> grow_basis_by_singles(
    const std::vector<std::bitset<nbits>> &seeds,
    const std::vector<double> &amplitudes,
    const std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>>
        &exclude,
    const std::vector<uint32_t> &as_orbs, size_t norbs,
    const GFSettings &settings) {
  assert(seeds.size() == amplitudes.size());
  std::vector<std::bitset<nbits>> found;
  std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>> found_pos;
  auto add = [&](const std::bitset<nbits> &det) {
    if(exclude.find(det) == exclude.end() &&
       found_pos.emplace(det, found.size()).second)
      found.push_back(det);
  };
  const auto within_cap = [&] {
    return settings.trunc_size == 0 || found.size() <= settings.trunc_size;
  };

  std::vector<std::bitset<nbits>> frontier;
  for(size_t k = 0; k < seeds.size(); ++k) {
    add(seeds[k]);
    if(std::abs(amplitudes[k]) >= settings.GFseedThres &&
       exclude.find(seeds[k]) == exclude.end())
      frontier.push_back(seeds[k]);
  }

  std::vector<std::bitset<nbits>> singles;
  for(int layer = 1; layer <= settings.tot_SD && within_cap(); ++layer) {
    const size_t start = found.size();
    for(const auto &det : frontier) {
      if(!within_cap()) break;
      generate_singles_spin_as(norbs, det, singles, as_orbs);
      for(const auto &s : singles) add(s);
    }
    frontier.assign(found.begin() + start, found.end());
  }
  return found;
}

/**
 * @brief Per-spin occupation of each of the first norbs spatial orbitals in
 *        wfn0, i.e. the diagonal of the spin-summed 1-RDM divided by 2, as
 *        evaluate_GF computes it for the GF active space.
 */
template <size_t nbits>
std::vector<double> orbital_occupations(
    const Eigen::VectorXd &wfn0, const std::vector<std::bitset<nbits>> &dets,
    size_t norbs) {
  assert(wfn0.size() == Eigen::Index(dets.size()));
  assert(norbs <= nbits / 2);
  std::vector<double> occs(norbs, 0.0);
  for(Eigen::Index k = 0; k < wfn0.size(); ++k) {
    const double w = wfn0[k] * wfn0[k];
    for(size_t i = 0; i < norbs; ++i)
      occs[i] +=
          w * (double(dets[k].test(i)) + double(dets[k].test(i + nbits / 2)));
  }
  const double norm = 2.0 * wfn0.squaredNorm();
  if(norm > 0.0)
    for(auto &o : occs) o /= norm;
  return occs;
}

/**
 * @brief Splits `added` by H-connectivity, with couplings those a CSR build
 *        with threshold h_thresh keeps (with the resolvent's threshold, the
 *        Lanczos Hamiltonian's nonzero elements):
 *
 *        - -1: H couples the determinant to `base`, directly or through other
 *          determinants of `added` (the H-connected component of base);
 *        - c >= 0: the determinant belongs to the c-th connected component of
 *          the rest, numbered in order of first appearance in `added`.
 *
 *        Different labels have no matrix element between them.
 */
template <size_t nbits, typename index_t = int32_t>
std::vector<int> added_components(const std::vector<std::bitset<nbits>> &base,
                                  const std::vector<std::bitset<nbits>> &added,
                                  HamiltonianGenerator<nbits> &Hgen,
                                  double h_thresh) {
  const size_t na = added.size();
  constexpr int unset = std::numeric_limits<int>::min();
  std::vector<int> label(na, unset);
  if(na == 0) return label;
  // Columns: added, then base. Rows: added, i.e. the first na columns. The
  // added block comes first because SDBuildHamiltonianGenerator stores the
  // diagonal element of row i at column i (it assumes a square tile); this
  // order puts it on the row's own determinant. H is symmetric, so each row
  // also lists the added neighbours that couple back to it.
  std::vector<std::bitset<nbits>> all(added);
  all.insert(all.end(), base.begin(), base.end());
  const auto H = make_csr_hamiltonian_block<index_t>(
      all.begin(), all.begin() + na, all.begin(), all.end(), Hgen, h_thresh);
  const auto &rowptr = H.rowptr();
  const auto &colind = H.colind();

  // Flood-fill from a set of start rows, labelling every reachable added row.
  std::vector<size_t> stack;
  auto flood = [&](int value) {
    while(!stack.empty()) {
      const size_t i = stack.back();
      stack.pop_back();
      for(auto p = rowptr[i]; p < rowptr[i + 1]; ++p) {
        const size_t j = colind[p];
        if(j >= na || label[j] != unset) continue;
        label[j] = value;
        stack.push_back(j);
      }
    }
  };

  // psi0's component: rows with a matrix element into base.
  for(size_t i = 0; i < na; ++i)
    for(auto p = rowptr[i]; p < rowptr[i + 1]; ++p)
      if(size_t(colind[p]) >= na) {
        label[i] = -1;
        stack.push_back(i);
        break;
      }
  flood(-1);

  // The remaining components.
  int ncomp = 0;
  for(size_t i = 0; i < na; ++i) {
    if(label[i] != unset) continue;
    label[i] = ncomp;
    stack.push_back(i);
    flood(ncomp);
    ++ncomp;
  }
  return label;
}

/**
 * @brief Flags the determinants of `added` that H couples to `base`, directly
 *        or through other determinants of `added` (added_components == -1).
 */
template <size_t nbits, typename index_t = int32_t>
std::vector<bool> coupled_to_base(const std::vector<std::bitset<nbits>> &base,
                                  const std::vector<std::bitset<nbits>> &added,
                                  HamiltonianGenerator<nbits> &Hgen,
                                  double h_thresh) {
  const auto label =
      added_components<nbits, index_t>(base, added, Hgen, h_thresh);
  std::vector<bool> coupled(label.size());
  for(size_t k = 0; k < label.size(); ++k) coupled[k] = label[k] == -1;
  return coupled;
}

namespace detail {

/**
 * @brief Adds to R the band-Lanczos matrix resolvent of the seed columns over
 *        `dets`: Gram matrix, deflation of its eigenvalues below
 *        orb_deflate_tol * lambda_max, BandResolvent on the orthonormalized
 *        retained seeds, back-transform to the npairs x npairs matrix per
 *        frequency. Returns the retained rank (0: nothing added).
 */
template <size_t nbits, typename index_t>
size_t add_block_resolvent(const Eigen::MatrixXd &seeds,
                           std::vector<std::bitset<nbits>> &dets,
                           HamiltonianGenerator<nbits> &Hgen, double E0,
                           const std::vector<std::complex<double>> &ws,
                           const GFSettings &settings, double h_thresh,
                           std::vector<std::vector<std::complex<double>>> &R) {
  const Eigen::Index npairs = seeds.cols();
  const Eigen::MatrixXd gram = seeds.transpose() * seeds;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(gram);
  if(eig.info() != Eigen::Success)
    throw std::runtime_error(
        "RunResolventOrbitalMatrix: Gram eigensolve failed");
  const double lambda_max =
      eig.eigenvalues().size() ? eig.eigenvalues().maxCoeff() : 0.0;
  if(lambda_max <= 0.0) return 0;

  std::vector<Eigen::Index> retained;
  for(Eigen::Index i = 0; i < eig.eigenvalues().size(); ++i)
    if(eig.eigenvalues()(i) > settings.orb_deflate_tol * lambda_max)
      retained.push_back(i);
  const size_t rank = retained.size();
  if(rank == 0) return 0;

  Eigen::MatrixXd Ur(npairs, rank);
  Eigen::VectorXd lambdas(rank);
  for(size_t i = 0; i < rank; ++i) {
    Ur.col(i) = eig.eigenvectors().col(retained[i]);
    lambdas(i) = eig.eigenvalues()(retained[i]);
  }
  Eigen::MatrixXd psi =
      seeds * Ur * lambdas.cwiseSqrt().cwiseInverse().asDiagonal();
  std::vector<double> vecs(psi.size());
  for(size_t i = 0; i < rank; ++i)
    for(Eigen::Index k = 0; k < psi.rows(); ++k)
      vecs[i * psi.rows() + k] = psi(k, i);

  auto hamil = make_dist_csr_hamiltonian<index_t>(MPI_COMM_WORLD, dets.begin(),
                                                  dets.end(), Hgen, h_thresh);
  // BandLan does not deflate: a seed's Krylov chain that is exhausted keeps
  // taking (zero) band slots while the other chains continue. A complete
  // calculation can therefore need up to rank * dets.size() iterations, not
  // dets.size(); capping at dets.size() truncated the longer chains on small
  // bases.
  const int nLanIts = int(std::min<size_t>(
      std::max<size_t>(settings.nLanIts, rank + 1), rank * dets.size()));
  std::vector<std::vector<std::complex<double>>> reduced;
  BandResolvent(hamil, vecs, ws, reduced, nLanIts, E0, true, rank, dets.size(),
                settings.print, settings.saveGFmats);

  const Eigen::MatrixXd B = Ur * lambdas.cwiseSqrt().asDiagonal();
  for(size_t iw = 0; iw < ws.size(); ++iw) {
    Eigen::MatrixXcd reduced_matrix(rank, rank);
    for(size_t k = 0; k < rank; ++k)
      for(size_t l = 0; l < rank; ++l)
        reduced_matrix(k, l) = reduced[iw][k * rank + l];
    const Eigen::MatrixXcd full = B * reduced_matrix * B.transpose();
    for(Eigen::Index k = 0; k < npairs; ++k)
      for(Eigen::Index l = 0; l < npairs; ++l)
        R[iw][k * npairs + l] += full(k, l);
  }
  return rank;
}

}  // namespace detail

struct OrbitalResolventResult {
  Eigen::MatrixXd gram;
  Eigen::VectorXd gram_eigenvalues;
  // Capture fraction of every seed on the input basis (the diagnostic).
  Eigen::VectorXd capture;
  // Capture fraction on the basis the resolvent was computed in. Equal to
  // capture unless the basis was expanded.
  Eigen::VectorXd capture_expanded;
  // Pairs marked for expansion (orb_expand_basis, capture below
  // orb_min_capture).
  std::vector<bool> expanded;
  size_t base_size = 0;
  size_t expanded_size = 0;
  // Expansion bookkeeping: leaked images used as growth seeds, determinants
  // grown (seeds included), and grown determinants dropped because H couples
  // them to base_dets (psi0's sector). expanded_size = base_size + grown -
  // dropped.
  size_t expansion_seeds = 0;
  size_t expansion_grown = 0;
  size_t expansion_dropped = 0;
  // True if the grown set exceeded settings.trunc_size, which stops the
  // growth (possibly before tot_SD layers, or with no growth at all when the
  // seeds alone exceed it).
  bool growth_capped = false;
  // Independent band-Lanczos runs: base_dets, plus one per connected
  // component of the kept determinants that some seed reaches.
  size_t lanczos_blocks = 0;
  // Sum of the retained ranks of all runs.
  size_t rank = 0;
  std::vector<std::vector<std::complex<double>>> resolvent;
};

/**
 * @brief Full n_imp^2 x n_imp^2 matrix resolvent of the orbital bilinears,
 *
 *          R_{mu nu; gamma delta}(w) =
 *              <phi_{mu nu}| 1 / (w - (H - E0)) |phi_{gamma delta}>,
 *          |phi_{mu nu}> = O_{mu nu}|wfn0>,
 *
 *        with O_{mu nu} = S_{mu nu} (Spin) or N_{mu nu} (Charge), see
 *        apply_orbital_bilinear. The seeds are Gram-deflated, passed to one
 *        band-Lanczos run and back-transformed. Pairs are indexed
 *        mu * n_imp + nu.
 *
 *        In the Charge channel <N_{mu mu}> is an orbital occupation, so the
 *        elastic pole m_k m_l / w dominates the diagonal block unless
 *        subtract_mean is set.
 *
 *        Basis expansion (settings.orb_expand_basis). A seed can leave the
 *        symmetry sector of base_dets (e.g. S_{mu nu}, mu != nu, when H
 *        conserves the electron count or parity of each orbital flavor), and
 *        then its capture fraction is 0 and its elements come out as zeros.
 *        With the flag set, every pair whose capture is below
 *        settings.orb_min_capture has its leaked images added to the basis,
 *        grown by grow_basis_by_singles (tot_SD, trunc_size, GFseedThres,
 *        asThres, norbs, as for the GF basis). A determinant leaked by several
 *        pairs is gated by its largest |amplitude|, so pairs cannot cancel.
 *
 *        The growth also produces determinants in wfn0's own sector that
 *        base_dets (a truncated ASCI space) does not contain. Kept, they would
 *        enlarge the space of the diagonal seeds and make wfn0 a non-eigenstate
 *        (poles below E0). So every grown determinant that H couples to
 *        base_dets, directly or through other grown determinants
 *        (coupled_to_base, same matrix-element threshold as the Lanczos
 *        Hamiltonian), is dropped. The kept determinants have no matrix
 *        element with base_dets, so the response of the unmarked pairs is the
 *        unexpanded one (to rounding). A pair that leaks only inside wfn0's
 *        sector (0 < capture < orb_min_capture) loses those images and stays
 *        below threshold: check capture_expanded.
 *
 *        All pairs are then evaluated on the single basis base_dets + kept,
 *        with wfn0 zero-padded. With the flag unset, or no pair below
 *        threshold, the result is identical to the unexpanded one.
 *
 * @param[in] DiagChannel channel: Spin (S_{mu nu}) or Charge (N_{mu nu}).
 * @param[in] bool subtract_mean: If true, use the fluctuation seeds
 *            (O_{mu nu} - <O_{mu nu}>)|wfn0>.
 * @param[in] n_active: Number of active orbitals, used as norbs for the
 *            growth when settings.norbs is 0. The active space of the growth
 *            is set by the per-spin occupations of wfn0
 *            (orbital_occupations, active_space_orbitals with asThres).
 */
template <size_t nbits, typename index_t = int32_t>
OrbitalResolventResult RunResolventOrbitalMatrix(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets, size_t n_imp,
    DiagChannel channel, double E0, const std::vector<std::complex<double>> &ws,
    const GFSettings &settings, bool subtract_mean = false,
    size_t n_active = 0) {
  // Matrix elements below this are dropped from the Lanczos Hamiltonian; the
  // sector filter of the expansion uses the same threshold.
  constexpr double h_thresh = 1.E-6;
  if(n_imp > nbits / 2)
    throw std::runtime_error(
        "RunResolventOrbitalMatrix: n_imp exceeds the spatial-orbital "
        "capacity");
  if(settings.orb_deflate_tol < 0.0)
    throw std::runtime_error(
        "RunResolventOrbitalMatrix: orb_deflate_tol must be non-negative");

  const size_t npairs = n_imp * n_imp;
  std::map<std::bitset<nbits>, size_t, bitset_less_comparator<nbits>> det_index;
  for(size_t k = 0; k < base_dets.size(); ++k)
    det_index.emplace(base_dets[k], k);

  // Capture pass on the input basis. Leaked images of the pairs below
  // threshold are merged with max |amplitude| (not summed: images of
  // different operators must not interfere).
  OrbitalResolventResult result;
  result.capture.resize(npairs);
  result.expanded.assign(npairs, false);
  std::map<std::bitset<nbits>, double, bitset_less_comparator<nbits>> leaked;
  for(size_t mu = 0; mu < n_imp; ++mu)
    for(size_t nu = 0; nu < n_imp; ++nu) {
      const size_t pair = mu * n_imp + nu;
      auto image =
          orbital_bilinear_image(wfn0, base_dets, det_index, mu, nu, channel);
      result.capture(pair) = image.capture;
      if(!settings.orb_expand_basis ||
         image.capture >= settings.orb_min_capture)
        continue;
      result.expanded[pair] = true;
      merge_leaked_images(leaked, image);
    }

  // The basis the resolvent is computed in: base_dets, followed by the grown
  // determinants if any pair was marked.
  std::vector<std::bitset<nbits>> dets(base_dets);
  Eigen::VectorXd psi0 = wfn0;
  // Block of each determinant: 0 for base_dets, 1 + c for the kept grown
  // determinants of connected component c. H has no matrix element between
  // blocks.
  std::vector<size_t> block_of(base_dets.size(), 0);
  size_t nblocks = 1;
  if(!leaked.empty()) {
    const size_t norbs = settings.norbs ? settings.norbs : n_active;
    if(norbs == 0 || norbs > nbits / 2)
      throw std::runtime_error(
          "RunResolventOrbitalMatrix: basis expansion needs settings.norbs "
          "or n_active, within the spatial-orbital capacity");
    const auto as_orbs = active_space_orbitals(
        orbital_occupations(wfn0, base_dets, norbs), settings.asThres);

    std::vector<std::bitset<nbits>> grow_seeds;
    std::vector<double> grow_amplitudes;
    for(const auto &[det, amp] : leaked) {
      grow_seeds.push_back(det);
      grow_amplitudes.push_back(amp);
    }
    const auto grown = grow_basis_by_singles(
        grow_seeds, grow_amplitudes, det_index, as_orbs, norbs, settings);
    result.expansion_seeds = grow_seeds.size();
    result.expansion_grown = grown.size();
    result.growth_capped = settings.trunc_size > 0 && settings.tot_SD > 0 &&
                           grown.size() > settings.trunc_size;

    // Keep only the grown determinants outside psi0's H-connected sector,
    // and remember the connected component each kept one belongs to.
    const auto label =
        added_components<nbits, index_t>(base_dets, grown, Hgen, h_thresh);
    for(size_t k = 0; k < grown.size(); ++k) {
      if(label[k] < 0) {
        ++result.expansion_dropped;
        continue;
      }
      det_index.emplace(grown[k], dets.size());
      dets.push_back(grown[k]);
      block_of.push_back(1 + label[k]);
      nblocks = std::max(nblocks, size_t(2 + label[k]));
    }
    psi0.conservativeResize(dets.size());
    psi0.tail(dets.size() - base_dets.size()).setZero();
  }
  result.base_size = base_dets.size();
  result.expanded_size = dets.size();

  Eigen::MatrixXd seeds(dets.size(), npairs);
  result.capture_expanded = result.capture;
  for(size_t mu = 0; mu < n_imp; ++mu)
    for(size_t nu = 0; nu < n_imp; ++nu) {
      const size_t pair = mu * n_imp + nu;
      seeds.col(pair) =
          apply_orbital_bilinear(psi0, dets, det_index, mu, nu, channel);
      if(dets.size() == base_dets.size()) continue;
      result.capture_expanded(pair) = orbital_bilinear_captured_fraction(
          psi0, dets, det_index, mu, nu, channel);
    }

  // H is block diagonal over the blocks, so the resolvent is the sum of the
  // resolvents of the seeds' restrictions to each block, each run on its own
  // Hamiltonian. Separate runs keep a small block (e.g. a truncated psi0
  // sector) from being exhausted inside a longer band-Lanczos run, which
  // has no deflation or reorthogonalization. Block 0 is base_dets alone, so
  // the elements of the unexpanded pairs are exactly the gate-off ones.
  result.gram = Eigen::MatrixXd::Zero(npairs, npairs);
  result.resolvent.assign(ws.size(), std::vector<std::complex<double>>(
                                         npairs * npairs, {0.0, 0.0}));
  std::vector<std::vector<Eigen::Index>> block_rows(nblocks);
  for(size_t k = 0; k < dets.size(); ++k) block_rows[block_of[k]].push_back(k);
  for(size_t b = 0; b < nblocks; ++b) {
    const auto &rows = block_rows[b];
    Eigen::MatrixXd block_seeds(rows.size(), npairs);
    std::vector<std::bitset<nbits>> block_dets(rows.size());
    for(size_t r = 0; r < rows.size(); ++r) {
      block_seeds.row(r) = seeds.row(rows[r]);
      block_dets[r] = dets[rows[r]];
    }

    // Optionally replace each seed O_{mu nu}|wfn0> by the fluctuation
    // (O_{mu nu} - <O_{mu nu}>)|wfn0>, cancelling the elastic pole exactly as
    // in RunResolventDiagonal. wfn0 lies in base_dets (block 0), so the
    // projected seed still gives the exact <O_{mu nu}>. The Gram matrix then
    // becomes the fluctuation covariance and remains the zeroth moment of R.
    // The capture fractions above describe the bare operator: the subtracted
    // component is in-basis.
    if(subtract_mean && b == 0)
      block_seeds -=
          wfn0 * ((wfn0.transpose() * block_seeds) / wfn0.squaredNorm());

    // A block no seed reaches contributes nothing: skip its CSR build.
    if(b > 0 && block_seeds.squaredNorm() == 0.0) continue;
    result.gram += block_seeds.transpose() * block_seeds;
    const size_t block_rank = detail::add_block_resolvent<nbits, index_t>(
        block_seeds, block_dets, Hgen, E0, ws, settings, h_thresh,
        result.resolvent);
    result.rank += block_rank;
    if(block_rank > 0) ++result.lanczos_blocks;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(result.gram);
  if(eig.info() != Eigen::Success)
    throw std::runtime_error(
        "RunResolventOrbitalMatrix: Gram eigensolve failed");
  result.gram_eigenvalues = eig.eigenvalues();
  return result;
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
  return weighted_imp_value<nbits>(det, w, DiagChannel::Spin, n_imp, n_active);
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
 *        With subtract_mean = true, the operator is replaced by its
 *        fluctuation delta_O = O - <O> (<O> = <wfn0|O|wfn0>/<wfn0|wfn0>) and
 *        the resolvent of the modified start vector
 *
 *          delta_v = O |wfn0> - <O> |wfn0>
 *
 *        is evaluated instead. For a reference state that is an eigenstate of
 *        H, expanding (O - <O>) G (O - <O>) with G = 1/(w - (H - E0)) and
 *        G |wfn0> = |wfn0>/w cancels the elastic (n = 0) pole exactly, so
 *
 *          R_delta(w) = sum_{n>0} |<0|O|n>|^2 / (w - w_n)
 *
 *        is the purely inelastic Lehmann sum. This is the same continued
 *        fraction with a different start vector, not an extra correction
 *        term; no change to the Lanczos routine is involved.
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
 * @param[in] bool subtract_mean: If true, evaluate the resolvent of the
 *            fluctuation delta_O = O - <O> instead of O (see above).
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
    const GFSettings &settings, bool subtract_mean = false) {
  // decompose_det packs the impurity occupation into a uint64_t (n_imp bits
  // per spin), so 2 * n_imp must fit in 64 bits.
  if(2 * n_imp > 64)
    throw std::runtime_error(
        "RunResolventDiagonal: 2*n_imp > 64 not supported");

  // Build v = O |wfn0> by rescaling each determinant coefficient.
  Eigen::VectorXd v =
      apply_diagonal_operator<nbits>(wfn0, base_dets, scalar_fn);

  // Optionally subtract the reference expectation <O> so that v becomes
  // delta_O |wfn0>. The denominator is <wfn0|wfn0> (not 1) because wfn0 need
  // not be normalized, consistently with the rest of this header.
  if(subtract_mean) {
    const double Omean = wfn0.dot(v) / wfn0.squaredNorm();
    v -= Omean * wfn0;
  }

  // If O |wfn0> (or delta_O |wfn0>) vanishes (e.g. Sz_imp on an Sz_tot = 0
  // state with n_imp == n_active, any traceless operator on an orbitally
  // unpolarized determinant set, or O proportional to the identity when
  // subtract_mean is set), the resolvent is identically zero. Return early to
  // avoid feeding a zero start vector into the Lanczos routine (which would
  // divide by ||v||).
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
 *        With subtract_mean = true, the fluctuation
 *        delta_Sz = Sz_imp - <Sz_imp> is used instead, dropping the elastic
 *        pole (see RunResolventDiagonal).
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
 * @param[in] bool subtract_mean: If true, evaluate the fluctuation resolvent
 *            of Sz_imp - <Sz_imp>.
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
    const GFSettings &settings, bool subtract_mean = false) {
  const auto sz_operator = [&](const std::bitset<nbits> &d) {
    return sz_imp_value<nbits>(d, n_imp, n_active);
  };
  return RunResolventDiagonal<nbits, index_t>(wfn0, Hgen, base_dets,
                                              sz_operator, n_imp, E0, ws,
                                              settings, subtract_mean);
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
 *        (RunResolventSz is kept as a separate entry point for that case).
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
 * @param[in] bool subtract_mean: If true, evaluate the fluctuation resolvent
 *            of O - <O> instead of O (see RunResolventDiagonal).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 *
 * @date 18/09/2026
 */
template <size_t nbits, typename index_t = int32_t>
std::vector<std::complex<double>> RunResolventWeighted(
    const Eigen::VectorXd &wfn0, HamiltonianGenerator<nbits> &Hgen,
    const std::vector<std::bitset<nbits>> &base_dets,
    const std::vector<double> &w, DiagChannel ch, size_t n_imp, size_t n_active,
    double E0, const std::vector<std::complex<double>> &ws,
    const GFSettings &settings, bool subtract_mean = false) {
  const auto op = [&](const std::bitset<nbits> &d) {
    return weighted_imp_value<nbits>(d, w, ch, n_imp, n_active);
  };
  return RunResolventDiagonal<nbits, index_t>(wfn0, Hgen, base_dets, op, n_imp,
                                              E0, ws, settings, subtract_mean);
}

}  // namespace macis
