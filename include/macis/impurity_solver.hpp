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

  if(todelete_h != todelete_p)
    throw std::runtime_error(
        "In evaluate_GF: todelete_h != todelete_p, the particle and hole "
        "Green's functions dropped different orbitals and cannot be summed");

  GF = macis::sum_GFs(GF, GF_tmp, ws, gf_settings.GF_orbs_comp, todelete_p);

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

  // The back-rotation also assumes GF row j corresponds to impurity orbital j.
  // That mapping breaks as soon as an orbital is dropped, since the surviving
  // rows are then GF_orbs_comp minus todelete_p.
  if(!todelete_p.empty())
    throw std::runtime_error(
        "In evaluate_GF: orbitals were dropped from the Green's function, so "
        "its rows no longer map onto impurity orbitals 0..n_imp-1 and the "
        "back-rotation would mix the wrong entries.");

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
    macis::write_GF(GF, ws, gf_settings.GF_orbs_comp, todelete_p);

  return GF;
}

/**
 * @brief Evaluates the dynamical impurity Sz-Sz response, i.e. the retarded
 *        resolvent of the impurity Sz operator on the ASCI ground state:
 *
 *          R(w) = <psi0| Sz_imp  1/(w - (H - E0))  Sz_imp |psi0>
 *
 *        Sz_imp is applied to the ground state by rescaling each determinant
 *        coefficient (see macis::RunResolventSz / macis::sz_imp_value), and the
 *        single-vector resolvent of the resulting state is evaluated over the
 *        same frequency grid used by evaluate_GF. The result R(w) is written to
 *        "Sz_resolvent.dat" (columns: Re(w) Im(w) Re(R) Im(R)) when
 *        gf_settings.writeGF_singlef is set, and returned to the caller.
 *
 * @param[in] double EASCI: ASCI ground-state energy (including core/inactive).
 * @param[in] macis::impurity_params<N> &p: Impurity problem parameters.
 * @param[in] macis::SDBuildHamiltonianGenerator<N> &ham_gen: Hamiltonian
 *            generator.
 * @param[in] macis::GFSettings &gf_settings: GF/resolvent settings (frequency
 *            grid, nLanIts, saveGFmats, writeGF_singlef).
 *
 * @returns std::vector<std::complex<double>>: R(w) along the frequency grid.
 */
template <size_t N>
auto evaluate_resolvent_sz(double EASCI, macis::impurity_params<N> &p,
                           macis::SDBuildHamiltonianGenerator<N> &ham_gen,
                           macis::GFSettings &gf_settings) {
  Eigen::VectorXd psi0 =
      Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(p.C.data(), p.C.size());

  std::cout << "EASCI = " << EASCI << std::endl;

  // Frequency grid (same convention as evaluate_GF).
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

  // Reference energy relative to the active-space Hamiltonian (matching the
  // shift used for the Green's function in evaluate_GF).
  double E0 = EASCI - (p.E_core + p.E_inactive);

  std::vector<std::complex<double>> R = macis::RunResolventSz<N>(
      psi0, ham_gen, p.dets, p.n_imp, p.n_active, E0, ws, gf_settings);

  if(gf_settings.writeGF_singlef) {
    using dbl = std::numeric_limits<double>;
    std::ofstream ofile("Sz_resolvent.dat");
    ofile.precision(dbl::max_digits10);
    for(size_t iii = 0; iii < ws.size(); iii++)
      ofile << std::scientific << real(ws[iii]) << " " << imag(ws[iii]) << " "
            << real(R[iii]) << " " << imag(R[iii]) << std::endl;
  }

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
