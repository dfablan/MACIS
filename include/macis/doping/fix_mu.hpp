/**
 * @file fix_mu.hpp
 * @brief Implement structures and routines to optimize the chemical potential
 * mu in an impurity problem to fix the number of particles.
 *
 */

#pragma once
// #ifndef MACIS_DOPING_FIX_MU_HPP
// #define MACIS_DOPING_FIX_MU_HPP
#include <gsl/gsl_roots.h>
#include <gsl/gsl_vector.h>

#include <iostream>

#include "macis/doping/call_solver.hpp"

constexpr size_t nwfn_bits = 64;

namespace macis {

/**
 * @brief Structure to hold the parameters of the optimization problem.
 */

struct fix_mu_params {
  size_t* n_active;
  size_t* nbeta;
  size_t* nalpha;
  size_t* n_inactive;
  size_t* norb;
  double* nel;
  size_t* n_imp;

  double* abs_tol;
  size_t* maxiter;
  bool* print;
  double* init_shift;

  std::string* ci_exp;
  std::string* asci_wfn_fname;
  bool* compute_asci_E0;
  double* asci_E0;

  macis::MCSCFSettings* mcscf_settings;
  macis::ASCISettings* asci_settings;
  std::vector<double>* occs;
  std::vector<double>* C;
  std::vector<macis::wfn_t<nwfn_bits>>* dets;

  double* dstep;
  double* E_core;
  double* E;
  std::vector<double>* T;
  std::vector<double>* V;
  bool* just_singles;
};

/**
 * @brief Cost function to find the root for. The solution will optimize the
 * chemical potential mu to fix the number of particles.
 */

double Mu_Cost_f(double x, void* params);

/**
 * @brief Cost function to find the root for. The solution will optimize the
 * chemical potential mu to fix the number of particles.
 */

double Mu_Cost_df(double x, void* params);

// /**
// * @brief Cost function to find the root for. The solution will optimize the
// chemical potential mu to fix
// * the number of particles.
// */

void Mu_Cost_fdf(double x, void* params, double* f, double* df);

// /**
//  * @brief Simple output routine to print the header for the optimization
//  output.
//  *        Version with derivatives.
//  *
//  * @param[inout] ostream& stream: Output stream where the header will be
//  written.
//  *
//  */
void print_header_fix_mu_noder(std::ostream& stream);

// /**
//  * @brief Simple output routine to print the state of the optimization.
//  Version
//  *        with derivatives.
//  *
//  * @param[inout] ostream& stream: Output stream where the state will be
//  broadcast.
//  * @param[in] size_t iter: Nr. of the iteration in the optimization process.
//  * @param[in] const gsl_multiroot_fdfsolver *s: Pointer to the multi-root
//  solver
//  *            being used for the optimization. This is a gradient solver.
//  */

void print_state_fix_mu_noder(std::ostream& stream, size_t iter,
                              const gsl_root_fdfsolver* s, double mu0);

// // /**
// //  * @brief Simple output routine to print the header for the optimization
// output.
// //  *        Version with no derivatives.
// //  *
// //  * @param[inout] ostream& stream: Output stream where the header will be
// written.
// //  *
// //  */
void print_header_fix_mu_noder(std::ostream& stream);

// // /**
// //  * @brief Simple output routine to print the state of the optimization.
// Version
// //  *        with no derivatives
// //  *
// //  * @param[inout] ostream& stream: Output stream where the state will be
// broadcast.
// //  * @param[in] size_t iter: Nr. of the iteration in the optimization
// process.
// //  * @param[in] const gsl_multiroot_fsolver *s: Pointer to the multi-root
// solver
// //  *            being used for the optimization. This is a gradient solver.
// //  *
// //  */
void print_state_fix_mu_noder(std::ostream& stream, size_t iter,
                              const gsl_root_fsolver* s, double mu0);

// /**
//  * @brief Actual optimization routine for the chemical potential. Calls GSL's
//  *        root finding functions (with derivative) to fix
//  *        mu to produce a given particle density.
//  *
//  * @param[in] const GhostGutzwiller &latt: Ghost gutzwiller class.
//  * @param[in] const double &init_mu: Initial chemical potential.
//  * @param[in] const double &ref_n: Reference particle density.
//  * @param[in] const Input_t &input: Dictionary with input parameters.
//  * @param[in] const std::vector<Eigen::MatrixXd>& Vs: Impurity
//  hybridizations.
//  * @param[in] const std::vector<Eigen::MatrixXd>& lambda_cs: Impurity bath
//  parameters.
//  * @param[in] const std::vector<Eigen::MatrixXd>& Chis: Chi tensors.
//  * @param[out] std::vector<Eigen::MatrixXd>& imp_1rdms: 1-RDMs of impurity
//  models with the
//  * @param[out] std::vector<Eigen::MatrixXd>& spsp_cfs: Spin-spin correlation
//  functions.
//  *             optimal chemical potential.
//  * @param[out] std::vector<double>& Eimps: Ground state energies of impurity
//  models with
//  *             the optimal chemical potential.
//  *
//  * @returns double: Optimal chemical potential
//  *
//  */
double Fix_Mu_der(const std::string& method_name, double& init_mu,
                  fix_mu_params* params);

// /**
//  * @brief Actual optimization routine for the chemical potential. Calls GSL's
//  *        root finding functions (no derivative) to fix
//  *        mu to produce a given particle density.
//  *
//  * @param[in] const GhostGutzwiller &latt: Ghost gutzwiller class.
//  * @param[in] const double &init_mu: Initial chemical potential.
//  * @param[in] const double &ref_n: Reference particle density.
//  * @param[in] const Input_t &input: Dictionary with input parameters.
//  * @param[in] const std::vector<Eigen::MatrixXd>& Vs: Impurity
//  hybridizations.
//  * @param[in] const std::vector<Eigen::MatrixXd>& lambda_cs: Impurity bath
//  parameters.
//  * @param[in] const std::vector<Eigen::MatrixXd>& Chis: Chi tensors.
//  * @param[out] std::vector<Eigen::MatrixXd>& imp_1rdms: 1-RDMs of impurity
//  models with the
//  * @param[out] std::vector<Eigen::MatrixXd>& spsp_cfs: Spin-spin correlation
//  functions.
//  *             optimal chemical potential.
//  * @param[out] std::vector<double>& Eimps: Ground state energies of impurity
//  models with
//  *             the optimal chemical potential.
//  *
//  * @returns double: Optimal chemical potential
//  *
double Fix_Mu_noder(const std::string& method_name, double& init_mu,
                    fix_mu_params* params);

}  // namespace macis
   // #endif // MACIS_DOPING_FIX_MU_HPP