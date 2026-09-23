#pragma once

#include <iomanip>
#include <iostream>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/gf/dynamical_properties.hpp>
#include <macis/gf/gf.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sd_build.hpp>
#include <macis/util/cas.hpp>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/util/general_io.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/moller_plesset.hpp>
#include <macis/util/mpi.hpp>
#include <macis/util/transform.hpp>
#include <macis/wavefunction_io.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>
#include <stdexcept>
#include <string>

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

enum class CIExpansion { CAS, ASCI, ASCI_cheap };

namespace macis {

/**
 * @brief Structure to hold the parameters of the impurity problem.
 */

template <size_t N>
struct impurity_params {
  size_t n_active;
  size_t nbeta;
  size_t nalpha;
  size_t n_inactive;
  size_t norb;
  size_t n_imp;
  size_t nbands;

  double nel_target;
  std::vector<double> orb_rot;

  double dstep;
  double abs_tol;
  size_t maxiter;
  size_t mu_cost_counter;
  bool print_doping;
  double init_shift;
  bool cheap_mode;
  double delta_CFS;
  bool spin_dep;

  CIExpansion ci_exp;
  std::string asci_wfn_fname;
  bool compute_asci_E0;
  double asci_E0;

  macis::MCSCFSettings mcscf_settings;
  macis::ASCISettings asci_settings;
  std::vector<double> occs;
  std::vector<double> C;
  std::vector<macis::wfn_t<N>> dets;

  double E_core;
  double E;
  double E_inactive;
  std::vector<double> T;
  std::vector<double> Td;
  std::vector<double> V;
  std::vector<double> F_inactive;
  std::vector<double> Fd_inactive;
  std::vector<double> T_active;
  std::vector<double> Td_active;
  std::vector<double> V_active;
  bool just_singles;
};

template <size_t N>
auto evaluate_GF(double EASCI, macis::impurity_params<N> &p,
                 macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                 macis::GFSettings &gf_settings) {
  Eigen::VectorXd psi0 =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p.C.data(), p.C.size());

  std::vector<double> active_ordm(p.n_active * p.n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  ham_gen.form_rdms(
      p.dets.begin(), p.dets.end(), p.dets.begin(), p.dets.end(), p.C.data(),
      macis::matrix_span<double>(active_ordm.data(), p.n_active, p.n_active),
      macis::rank4_span<double>(active_trdm.data(), p.n_active, p.n_active,
                                p.n_active, p.n_active));

  std::vector<double> occs(p.n_active, 0.);
  for(int i = 0; i < p.n_active; i++)
    occs[i] += active_ordm[i + i * p.n_active] / 2.;

  std::cout << "Orbital Occupations in the rotated basis (nrots = "
            << p.asci_settings.nrots << "):" << std::endl;
  std::cout << "(rotated)Occs: ";
  for(const auto oc : occs) std::cout << oc << ", ";
  std::cout << std::endl;

  std::cout << "EASCI = " << EASCI << std::endl;

  // Frequency grid
  std::vector<std::complex<double>> ws(gf_settings.nws,
                                       std::complex<double>(0., 0.));

  for(int i = 0; i < gf_settings.nws; i++)
    if(gf_settings.imag_freq) {
      //  MATSUBARA GRID
      ws[i] = std::complex<double>(0., (2 * i + 1) * M_PI / gf_settings.beta);
    } else {
      std::complex<double> w0(gf_settings.wmin, gf_settings.eta);
      std::complex<double> wf(gf_settings.wmax, gf_settings.eta);
      ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);
    }

  // GF vector
  std::vector<std::vector<std::complex<double>>> GF(
      gf_settings.nws,
      std::vector<std::complex<double>>(p.n_active * p.n_active,
                                        std::complex<double>(0., 0.)));
  std::vector<std::vector<std::complex<double>>> GF_tmp(
      gf_settings.nws,
      std::vector<std::complex<double>>(p.n_active * p.n_active,
                                        std::complex<double>(0., 0.)));

  // GS vector
  std::vector<int> todelete_p;
  std::vector<int> todelete_h;

  EASCI -= (p.E_core + p.E_inactive);
  // Evaluate particle GF
  macis::RunGFCalc<N>(GF_tmp, psi0, ham_gen, p.dets, EASCI, true, ws, occs,
                      gf_settings, todelete_p);

  // std::cout << "GF Particle part calculated." << std::endl;
  // for(int i = 0; i < p.n_imp; i++) {
  //   for(int j = 0; j < p.n_imp; j++)
  //     std::cout << GF_tmp[0][i + j * p.n_active] << " " << std::endl;
  // }

  // Evaluate hole GF
  macis::RunGFCalc<N>(GF, psi0, ham_gen, p.dets, EASCI, false, ws, occs,
                      gf_settings, todelete_h);

  EASCI += p.E_core + p.E_inactive;

  // Both sectors now return full GF_orbs_comp^2 matrices: RunGFCalc pads the
  // dropped (vanishing add/remove vector) rows/cols with zeros. They may have
  // dropped different orbitals (e.g. a fully occupied orbital is dropped from
  // the particle sector while an empty one is dropped from the hole sector),
  // but each sector's contribution to a dropped orbital is zero, so the two
  // matrices can simply be added elementwise.
  for(size_t iw = 0; iw < gf_settings.nws; iw++)
    for(size_t k = 0; k < GF[iw].size(); k++) GF[iw][k] += GF_tmp[iw][k];

  // std::cout << "GF hole part calculated." << std::endl;
  // for(int i = 0; i < p.n_imp; i++) {
  //   for(int j = 0; j < p.n_imp; j++)
  //     std::cout << GF[0][i + j * p.n_active] << " " << std::endl;
  // }

  // Rotate the GF back to original basis

  size_t G_n_orbs = sqrt(GF[0].size());

  // The loops below index rows/columns 0..n_imp-1 of the returned GF, so the
  // GF must be at least that large. It is not if fewer than n_imp orbitals
  // were requested in GF_orbs_comp, or if any of them were dropped.
  if(G_n_orbs < p.n_imp)
    throw std::runtime_error(
        "In evaluate_GF: the computed Green's function is smaller than n_imp, "
        "cannot rotate the impurity block back to the original basis. Check "
        "GF.ORBS_COMP and the list of dropped orbitals.");

  // The back-rotation assumes GF row j corresponds to impurity orbital j. That
  // mapping now always holds: RunGFCalc pads the GF back to the full
  // GF_orbs_comp index space (with zeros where an orbital's add/remove vector
  // vanished), so rows map onto GF_orbs_comp order, and hence onto impurity
  // orbitals 0..n_imp-1 whenever GF_orbs_comp is the full impurity set.
  // A zero row/col here means that orbital contributes nothing from the
  // corresponding sector, which is the correct physical limit.

  Eigen::MatrixXd rotMat = Eigen::MatrixXd::Identity(p.n_imp, p.n_imp);
  for(int j = 0; j < p.n_imp; j++)
    for(int k = 0; k < p.n_imp; k++)
      rotMat(j, k) = p.orb_rot[j + k * p.n_active];

  // rotMat is the top-left n_imp x n_imp block of the n_active x n_active
  // orb_rot. That block is only unitary on its own if orb_rot is block-diagonal
  // in imp/bath, which rotate_hamiltonian_ordm_imp_bath guarantees but the full
  // rotate_hamiltonian_ordm (asci_grow's grow_with_rot path) does not. Without
  // this check a non-block-diagonal orb_rot would silently apply a
  // non-unitary truncation to the GF.
  if(p.n_imp > 0) {
    const Eigen::MatrixXd dev = rotMat * rotMat.transpose() -
                                Eigen::MatrixXd::Identity(p.n_imp, p.n_imp);
    const double unitarity_tol = 1.e-8;
    if(dev.cwiseAbs().maxCoeff() > unitarity_tol)
      throw std::runtime_error(
          "In evaluate_GF: the impurity block of orb_rot is not unitary (max "
          "deviation of R R^T from the identity is " +
          std::to_string(dev.cwiseAbs().maxCoeff()) +
          "). orb_rot is not block-diagonal in imp/bath, so the impurity "
          "Green's function cannot be rotated back with its impurity block "
          "alone.");
  }

  for(int iw = 0; iw < gf_settings.nws; iw++) {
    Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(p.n_imp, p.n_imp);
    for(int j = 0; j < p.n_imp; j++)
      for(int k = 0; k < p.n_imp; k++) {
        G(j, k) = GF[iw][j + k * G_n_orbs];
      }

    //  Eigen::MatrixXcd rotG  = rotMat.adjoint() * G * rotMat;
    Eigen::MatrixXcd rotG = rotMat * G * rotMat.adjoint();

    for(int j = 0; j < p.n_imp; j++)
      for(int k = 0; k < p.n_imp; k++) GF[iw][j + k * G_n_orbs] = rotG(j, k);
  }

  if(gf_settings.writeGF_singlef)
    macis::write_GF(GF, ws, gf_settings.GF_orbs_comp, std::vector<int>{});

  return GF;
}

namespace detail {

// Frequency grid shared by evaluate_resolvent_sz and
// evaluate_resolvent_diagonal: real axis as in evaluate_GF, but on the
// imaginary axis a BOSONIC Matsubara grid (even multiples of pi/beta), since
// every diagonal-operator resolvent in this header (Sz-Sz, orbital, staggered
// spin, ...) is a bosonic correlator, unlike the fermionic GF which lives on
// odd multiples (2*i+1)*pi/beta.
//
// The grid starts at nu_1 = 2*pi/beta, NOT at nu_0 = 0. w = 0 is a point on
// the REAL axis, which is exactly where the poles of the continued fraction
// sit, and there is no i*eta broadening on the Matsubara axis to keep it away
// from them. E0 comes from Davidson at CI_RES_TOL, while the smallest
// eigenvalue of the Lanczos tridiagonal matrix is the exact ground state of H
// in the Krylov space, so the two differ by O(CI_RES_TOL) with an essentially
// arbitrary sign. That plants a spurious pole at w ~ delta -> 0 whose
// contribution to R is -weight/delta at w = 0 and only
// -weight*delta/(nu^2 + delta^2) ~ 0 at every nu >> delta. The nu = 0 value
// was therefore the only unusable point on the grid (observed: static
// susceptibilities wrong by orders of magnitude, and negative), while
// nu >= nu_1 is smooth. Recover the static limit by extrapolating R(i*nu)
// from the lowest few Matsubara points instead.
inline std::vector<std::complex<double>> build_bosonic_resolvent_grid(
    const macis::GFSettings &gf_settings) {
  std::vector<std::complex<double>> ws(gf_settings.nws,
                                       std::complex<double>(0., 0.));
  for(int i = 0; i < gf_settings.nws; i++)
    if(gf_settings.imag_freq) {
      //  BOSONIC MATSUBARA GRID, STARTING AT nu_1 = 2*pi/beta
      ws[i] = std::complex<double>(
          0., 2. * double(i + 1) * M_PI / gf_settings.beta);
    } else {
      std::complex<double> w0(gf_settings.wmin, gf_settings.eta);
      std::complex<double> wf(gf_settings.wmax, gf_settings.eta);
      ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);
    }
  return ws;
}

// Writes a diagonal-operator resolvent to "<label>_resolvent.dat" (columns:
// Re(w) Im(w) Re(R) Im(R)), guarding the write so only rank 0 touches the
// shared filename (all ranks compute identical R(w), so one write suffices).
inline void write_resolvent_singlef(
    const std::string &label, const std::vector<std::complex<double>> &ws,
    const std::vector<std::complex<double>> &R) {
  bool write_file = true;
  MACIS_MPI_CODE(write_file = (macis::comm_rank(MPI_COMM_WORLD) == 0);)
  if(write_file) {
    using dbl = std::numeric_limits<double>;
    std::ofstream ofile(label + "_resolvent.dat");
    ofile.precision(dbl::max_digits10);
    for(size_t iii = 0; iii < ws.size(); iii++)
      ofile << std::scientific << real(ws[iii]) << " " << imag(ws[iii])
            << " " << real(R[iii]) << " " << imag(R[iii]) << std::endl;
  }
}

}  // namespace detail

/**
 * @brief Evaluates the dynamical impurity Sz-Sz response, i.e. the retarded
 *        resolvent of the impurity Sz operator on the ASCI ground state:
 *
 *          R(w) = <psi0| Sz_imp  1/(w - (H - E0))  Sz_imp |psi0>
 *
 *        Sz_imp is applied to the ground state by rescaling each determinant
 *        coefficient (see macis::RunResolventSz / macis::sz_imp_value), and the
 *        single-vector resolvent of the resulting state is evaluated over the
 *        same real grid as evaluate_GF but on a bosonic Matsubara grid
 *        (even multiples of pi/beta) on the imaginary axis, since the Sz-Sz
 *        response is a bosonic correlator. The grid starts at nu_1 = 2*pi/beta
 *        and deliberately EXCLUDES nu_0 = 0: see build_bosonic_resolvent_grid.
 *        The result R(w) is written to "Sz_resolvent.dat" (columns: Re(w)
 *        Im(w) Re(R) Im(R)) when gf_settings.writeGF_singlef is set, and
 *        returned to the caller.
 *
 *        Unlike evaluate_resolvent_diagonal, this does NOT require the
 *        impurity block of orb_rot to be the identity: Sz_imp sums uniformly
 *        over every impurity orbital and is invariant under the
 *        block-diagonal, spin-conserving natural-orbital rotations used by
 *        SolveImpurityASCI_rot (see sz_imp_value / §6 of
 *        PLAN_orbital_resolvent.md), so it remains valid on the rotated
 *        production path.
 *
 * @param[in] double EASCI: ASCI ground-state energy (including core/inactive).
 * @param[in] macis::impurity_params<N> &p: Impurity problem parameters.
 * @param[in] macis::SDBuildHamiltonianGenerator<N> &ham_gen: Hamiltonian
 *            generator.
 * @param[in] macis::GFSettings &gf_settings: GF/resolvent settings (frequency
 *            grid, nLanIts, saveGFmats, writeGF_singlef).
 * @param[in] bool subtract_mean: If true, evaluate the resolvent of the
 *            fluctuation Sz_imp - <Sz_imp> instead of Sz_imp, dropping the
 *            elastic pole (see macis::RunResolventDiagonal).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 */
template <size_t N>
auto evaluate_resolvent_sz(double EASCI, macis::impurity_params<N> &p,
                           macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                           macis::GFSettings &gf_settings,
                           bool subtract_mean = false) {
  Eigen::VectorXd psi0 =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p.C.data(), p.C.size());

  std::cout << "EASCI = " << EASCI << std::endl;

  std::vector<std::complex<double>> ws =
      detail::build_bosonic_resolvent_grid(gf_settings);

  // Reference energy relative to the active-space Hamiltonian (matching the
  // shift used for the Green's function in evaluate_GF).
  double E0 = EASCI - (p.E_core + p.E_inactive);

  std::vector<std::complex<double>> R = macis::RunResolventSz<N>(
      psi0, ham_gen, p.dets, p.n_imp, p.n_active, E0, ws, gf_settings,
      subtract_mean);

  if(gf_settings.writeGF_singlef)
    detail::write_resolvent_singlef(subtract_mean ? "Sz_delta" : "Sz", ws, R);

  return R;
}

/**
 * @brief Evaluates the dynamical impurity response of a general
 *        per-orbital-weighted diagonal operator O on the ASCI ground state:
 *
 *          R(w) = <psi0| O  1/(w - (H - E0))  O |psi0>,
 *          O = macis::weighted_imp_value(., w, channel, n_imp, n_active)
 *
 *        Generalizes evaluate_resolvent_sz to an arbitrary diagonal impurity
 *        operator (orbital Cartan generators T^3/T^8, staggered spin, ...),
 *        specified by a per-impurity-orbital weight vector w and a channel
 *        (see macis::DiagChannel and the make_*_weights builders in
 *        dynamical_properties.hpp). Reuses the same frequency grid
 *        construction, E0 shift and output format as evaluate_resolvent_sz;
 *        the result is written to "<label>_resolvent.dat" when
 *        gf_settings.writeGF_singlef is set.
 *
 *        Unlike Sz_imp, an orbital- or site-resolved operator is NOT
 *        invariant under the natural-orbital rotation used by
 *        SolveImpurityASCI_rot: such a rotation scrambles exactly the
 *        orbital/site distinction these operators measure, and, because O is
 *        contracted on both sides of R(w), there is no surviving free index
 *        to rotate the result back through afterwards (unlike evaluate_GF's
 *        Green's function). O therefore has to be correct in the SAME basis
 *        as p.dets before it is applied, so this function throws unless the
 *        impurity block of p.orb_rot is the identity (see §6 of
 *        PLAN_orbital_resolvent.md).
 *
 * @param[in] double EASCI: ASCI ground-state energy (including core/inactive).
 * @param[in] macis::impurity_params<N> &p: Impurity problem parameters.
 * @param[in] macis::SDBuildHamiltonianGenerator<N> &ham_gen: Hamiltonian
 *            generator.
 * @param[in] macis::GFSettings &gf_settings: GF/resolvent settings (frequency
 *            grid, nLanIts, saveGFmats, writeGF_singlef).
 * @param[in] const std::vector<double> &w: Per-impurity-orbital weight, size
 *            p.n_imp.
 * @param[in] macis::DiagChannel channel: Charge or Spin channel.
 * @param[in] const std::string &label: Used to name the output file
 *            "<label>_resolvent.dat".
 * @param[in] bool subtract_mean: If true, evaluate the resolvent of the
 *            fluctuation O - <O> instead of O, dropping the elastic pole (see
 *            macis::RunResolventDiagonal).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 *
 * @throws std::runtime_error if the impurity block of p.orb_rot is not the
 *         identity within 1e-10.
 */
template <size_t N>
auto evaluate_resolvent_diagonal(double EASCI, macis::impurity_params<N> &p,
                                 macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                                 macis::GFSettings &gf_settings,
                                 const std::vector<double> &w,
                                 macis::DiagChannel channel,
                                 const std::string &label,
                                 bool subtract_mean = false) {
  // Rotation guard (§6 of PLAN_orbital_resolvent.md): the impurity block of
  // orb_rot must be the identity, not merely unitary (contrast evaluate_GF's
  // check above), because there is no free index left on R(w) to rotate an
  // orbital-resolved operator back through once it has been applied.
  if(p.n_imp > 0) {
    Eigen::MatrixXd rotBlock = Eigen::MatrixXd::Zero(p.n_imp, p.n_imp);
    for(int j = 0; j < p.n_imp; j++)
      for(int k = 0; k < p.n_imp; k++)
        rotBlock(j, k) = p.orb_rot[j + k * p.n_active];
    const Eigen::MatrixXd dev =
        rotBlock - Eigen::MatrixXd::Identity(p.n_imp, p.n_imp);
    const double identity_tol = 1.e-10;
    if(dev.cwiseAbs().maxCoeff() > identity_tol)
      throw std::runtime_error(
          "In evaluate_resolvent_diagonal (label = \"" + label +
          "\"): the impurity block of orb_rot is not the identity (max "
          "deviation from the identity is " +
          std::to_string(dev.cwiseAbs().maxCoeff()) +
          "). Orbital-resolved diagonal operators (orbital T^3/T^8, "
          "staggered spin, ...) are only meaningful in the unrotated "
          "impurity basis that p.dets is expressed in: a natural-orbital "
          "rotation scrambles exactly the orbital/site distinction they "
          "measure, and R(w) has no surviving free index to rotate back "
          "through afterwards. Set CI.EXPANSION = CAS or ASCI.NROTS = 0 to "
          "disable the rotation.");
  }

  Eigen::VectorXd psi0 =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p.C.data(), p.C.size());

  std::cout << "EASCI = " << EASCI << std::endl;

  std::vector<std::complex<double>> ws =
      detail::build_bosonic_resolvent_grid(gf_settings);

  double E0 = EASCI - (p.E_core + p.E_inactive);

  std::vector<std::complex<double>> R = macis::RunResolventWeighted<N>(
      psi0, ham_gen, p.dets, w, channel, p.n_imp, p.n_active, E0, ws,
      gf_settings, subtract_mean);

  if(gf_settings.writeGF_singlef)
    detail::write_resolvent_singlef(label, ws, R);

  return R;
}

template <size_t N>
auto evaluate_ordm(std::vector<macis::wfn_t<N>> &dets,
                   std::vector<double> &X_local,
                   macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                   std::vector<double> &orb_rot) {
  // Get Parameters
  size_t n_active = sqrt(orb_rot.size());

  // Compute the 1-rdm
  typename std::vector<macis::wfn_t<N>>::iterator det_st = dets.begin();
  typename std::vector<macis::wfn_t<N>>::iterator det_en = dets.end();

  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  ham_gen.form_rdms(
      dets.begin(), dets.end(), dets.begin(), dets.end(), X_local.data(),
      macis::matrix_span<double>(active_ordm.data(), n_active, n_active),
      macis::rank4_span<double>(active_trdm.data(), n_active, n_active,
                                n_active, n_active));

  std::cout << "Orbital Occupations in the rotated basis:" << std::endl;
  std::cout << "(rotated)Occs: ";
  for(int i = 0; i < n_active; i++)
    std::cout << active_ordm[i + i * n_active] / 2. << ", ";
  std::cout << std::endl;

  //   //print ordm DEBUG
  //  macis::util::write_matrix(active_ordm.data(), n_active, n_active,
  //                            "active_ordm_rotated.dat", true);

  // Rotate the 1-RDM back to original basis
  // Eigen::MatrixXd roto = orb_rot * o * orb_rot.adjoint();
  std::vector<double> tmp(n_active * n_active, 0.);
  std::vector<double> comp(n_active * n_active, 0.);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
             n_active, n_active, n_active, 1.0, active_ordm.data(), n_active,
             orb_rot.data(), n_active, 0.0, tmp.data(), n_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             n_active, n_active, n_active, 1.0, orb_rot.data(), n_active,
             tmp.data(), n_active, 0.0, comp.data(), n_active);

  return comp;
}

template <size_t N>
double SolveImpurityED(impurity_params<N> &params);

template <size_t N>
double SolveImpurityASCI(impurity_params<N> &params);

template <size_t N>
double SolveImpurityASCI_rot(impurity_params<N> &params);

template <size_t N>
double SolveImpurityCheapASCI(impurity_params<N> &params);

}  // namespace macis
