#pragma once

#include <iomanip>
#include <iostream>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
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
  std::vector<double> V;
  std::vector<double> F_inactive;
  std::vector<double> T_active;
  std::vector<double> Td_active;
  std::vector<double> V_active;
  bool just_singles;
};

template <size_t N>
auto evaluate_GF(const double EASCI, macis::impurity_params<N> &p,
                 macis::DoubleLoopHamiltonianGenerator<N> &ham_gen,
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
  std::cout << "Occs: ";
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

  // Evaluate particle GF
  macis::RunGFCalc<N>(GF_tmp, psi0, ham_gen, p.dets, EASCI, true, ws, occs,
                      gf_settings);

  std::cout << "GF Particle part calculated." << std::endl;
  for(int i = 0; i < p.n_imp; i++) {
    for(int j = 0; j < p.n_imp; j++)
      std::cout << GF_tmp[0][i + j * p.n_active] << " " << std::endl;
  }

  // Evaluate hole GF
  macis::RunGFCalc<N>(GF, psi0, ham_gen, p.dets, EASCI, false, ws, occs,
                      gf_settings);

  if(todelete_h != todelete_p)
    std::cout << "ERROR: todelete_h!=todelete_p" << std::endl;

  GF = macis::sum_GFs(GF, GF_tmp, ws, gf_settings.GF_orbs_comp, todelete_p);

  std::cout << "GF hole part calculated." << std::endl;
  for(int i = 0; i < p.n_imp; i++) {
    for(int j = 0; j < p.n_imp; j++)
      std::cout << GF[0][i + j * p.n_active] << " " << std::endl;
  }

  // Rotate the GF back to original basis

  size_t G_n_orbs = sqrt(GF[0].size());

  Eigen::MatrixXd rotMat = Eigen::MatrixXd::Identity(p.n_imp, p.n_imp);
  for(int j = 0; j < p.n_imp; j++)
    for(int k = 0; k < p.n_imp; k++)
      rotMat(j, k) = p.orb_rot[j + k * p.n_active];

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

template <size_t N>
auto evaluate_ordm(std::vector<macis::wfn_t<N>> &dets,
                   std::vector<double> &X_local,
                   macis::DoubleLoopHamiltonianGenerator<N> &ham_gen,
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