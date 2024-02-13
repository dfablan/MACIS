#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <iomanip>
#include <iostream>
#include <macis/doping/call_solver.hpp>
#include <macis/doping/fix_mu.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

#include "ini_input.hpp"

enum class CIExpansion { CAS, ASCI };

std::map<std::string, CIExpansion> ci_exp_map = {{"CAS", CIExpansion::CAS},
                                                 {"ASCI", CIExpansion::ASCI}};

template <typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

int main(int argc, char** argv) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  std::cout << std::scientific << std::setprecision(12);
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("[%n] %v");

  constexpr size_t nwfn_bits = 64;
  MACIS_MPI_CODE(MPI_Init(&argc, &argv);)

  std::vector<macis::wfn_t<nwfn_bits>> dets;
  std::vector<double> C;

  // Create Logger
  auto console = spdlog::stdout_color_mt("simple_driver");

  // Read Input Options
  std::vector<std::string> opts(argc);
  for(int i = 0; i < argc; ++i) opts[i] = argv[i];
  auto input_file = opts.at(1);
  INIFile input(input_file);
  // Required Keywords
  auto fcidump_fname = input.getData<std::string>("CI.FCIDUMP");
  auto nalpha = input.getData<size_t>("CI.NALPHA");
  auto nbeta = input.getData<size_t>("CI.NBETA");
  // if(nalpha != nbeta) throw std::runtime_error("NALPHA != NBETA");
  // Read FCIDUMP File
  size_t norb = macis::read_fcidump_norb(fcidump_fname);
  size_t norb2 = norb * norb;
  size_t norb3 = norb2 * norb;
  size_t norb4 = norb2 * norb2;
  std::vector<double> T(norb2), V(norb4);
  auto E_core = macis::read_fcidump_core(fcidump_fname);
  macis::read_fcidump_1body(fcidump_fname, T.data(), norb);
  macis::read_fcidump_2body(fcidump_fname, V.data(), norb);
  bool just_singles = macis::is_2body_diagonal(fcidump_fname);

#define OPT_KEYWORD(STR, RES, DTYPE) \
  if(input.containsData(STR)) {      \
    RES = input.getData<DTYPE>(STR); \
  }

  // Set up job

  std::string ciexp_str;
  OPT_KEYWORD("CI.EXPANSION", ciexp_str, std::string);
  CIExpansion ci_exp;
  try {
    ci_exp = ci_exp_map.at(ciexp_str);
  } catch(...) {
    throw std::runtime_error("CI Expansion Not Recognized");
  }

  // Set up active space
  size_t n_inactive = 0;
  OPT_KEYWORD("CI.NINACTIVE", n_inactive, size_t);
  if(n_inactive >= norb) throw std::runtime_error("NINACTIVE >= NORB");

  size_t n_active = norb - n_inactive;
  OPT_KEYWORD("CI.NACTIVE", n_active, size_t);

  if(n_inactive + n_active > norb)
    throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

  size_t n_virtual = norb - n_active - n_inactive;
  if(n_active > nwfn_bits / 2) throw std::runtime_error("Not Enough Bits");

  size_t n_imp = norb;
  OPT_KEYWORD("CI.NIMP", n_imp, size_t);

  // Misc optional files
  std::string rdm_fname, fci_out_fname;
  OPT_KEYWORD("CI.RDMFILE", rdm_fname, std::string);
  OPT_KEYWORD("CI.FCIDUMP_OUT", fci_out_fname, std::string);

  // MCSCF Settings
  macis::MCSCFSettings mcscf_settings;
  OPT_KEYWORD("MCSCF.MAX_MACRO_ITER", mcscf_settings.max_macro_iter, size_t);
  OPT_KEYWORD("MCSCF.MAX_ORB_STEP", mcscf_settings.max_orbital_step, double);
  OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL", mcscf_settings.orb_grad_tol_mcscf, double);
  // OPT_KEYWORD("MCSCF.BFGS_TOL",       mcscf_settings.orb_grad_tol_bfgs,
  // double); OPT_KEYWORD("MCSCF.BFGS_MAX_ITER", mcscf_settings.max_bfgs_iter,
  // size_t);
  OPT_KEYWORD("MCSCF.ENABLE_DIIS", mcscf_settings.enable_diis, bool);
  OPT_KEYWORD("MCSCF.DIIS_START_ITER", mcscf_settings.diis_start_iter, size_t);
  OPT_KEYWORD("MCSCF.DIIS_NKEEP", mcscf_settings.diis_nkeep, size_t);
  OPT_KEYWORD("MCSCF.CI_RES_TOL", mcscf_settings.ci_res_tol, double);
  OPT_KEYWORD("MCSCF.CI_MAX_SUB", mcscf_settings.ci_max_subspace, size_t);
  OPT_KEYWORD("MCSCF.CI_MATEL_TOL", mcscf_settings.ci_matel_tol, double);

  // ASCI Settings
  macis::ASCISettings asci_settings;
  std::string asci_wfn_fname, asci_wfn_out_fname;
  double asci_E0 = 0.0;
  bool compute_asci_E0 = true;
  OPT_KEYWORD("ASCI.NTDETS_MAX", asci_settings.ntdets_max, size_t);
  OPT_KEYWORD("ASCI.NTDETS_MIN", asci_settings.ntdets_min, size_t);
  OPT_KEYWORD("ASCI.NCDETS_MAX", asci_settings.ncdets_max, size_t);
  OPT_KEYWORD("ASCI.HAM_EL_TOL", asci_settings.h_el_tol, double);
  OPT_KEYWORD("ASCI.RV_PRUNE_TOL", asci_settings.rv_prune_tol, double);
  OPT_KEYWORD("ASCI.PAIR_MAX_LIM", asci_settings.pair_size_max, size_t);
  OPT_KEYWORD("ASCI.GROW_FACTOR", asci_settings.grow_factor, int);
  OPT_KEYWORD("ASCI.MAX_REFINE_ITER", asci_settings.max_refine_iter, size_t);
  OPT_KEYWORD("ASCI.REFINE_ETOL", asci_settings.refine_energy_tol, double);
  OPT_KEYWORD("ASCI.GROW_WITH_ROT", asci_settings.grow_with_rot, bool);
  OPT_KEYWORD("ASCI.ROT_SIZE_START", asci_settings.rot_size_start, size_t);
  // OPT_KEYWORD("ASCI.DIST_TRIP_RAND",  asci_settings.dist_triplet_random,
  // bool );
  OPT_KEYWORD("ASCI.CONSTRAINT_LVL", asci_settings.constraint_level, int);
  OPT_KEYWORD("ASCI.WFN_FILE", asci_wfn_fname, std::string);
  OPT_KEYWORD("ASCI.WFN_OUT_FILE", asci_wfn_out_fname, std::string);
  if(input.containsData("ASCI.E0_WFN")) {
    asci_E0 = input.getData<double>("ASCI.E0_WFN");
    compute_asci_E0 = false;
  }

  console->info("[Wavefunction Data]:");
  console->info("  * CIEXP   = {}", ciexp_str);
  console->info("  * FCIDUMP = {}", fcidump_fname);
  if(fci_out_fname.size()) console->info("  * FCIDUMP_OUT = {}", fci_out_fname);
  console->debug("READ {} 1-body integrals and {} 2-body integrals", T.size(),
                 V.size());
  console->info("ECORE = {:.12f}", E_core);
  console->debug("TSUM  = {:.12f}", vec_sum(T));
  console->debug("VSUM  = {:.12f}", vec_sum(V));
  console->info("TMEM   = {:.2e} GiB", macis::to_gib(T));
  console->info("VMEM   = {:.2e} GiB", macis::to_gib(V));

  // Setup printing
  bool print_davidson = true, print_ci = true, print_mcscf = true,
       print_diis = true, print_asci_search = true, print_determinants = true;
  double determinants_threshold = 1e-2;
  OPT_KEYWORD("PRINT.DAVIDSON", print_davidson, bool);
  OPT_KEYWORD("PRINT.CI", print_ci, bool);
  OPT_KEYWORD("PRINT.MCSCF", print_mcscf, bool);
  OPT_KEYWORD("PRINT.DIIS", print_diis, bool);
  OPT_KEYWORD("PRINT.ASCI_SEARCH", print_asci_search, bool);
  OPT_KEYWORD("PRINT.DETERMINANTS", print_determinants, bool);
  OPT_KEYWORD("PRINT.DETERMINANTS_THRES", determinants_threshold, double);
  if(not print_davidson) spdlog::null_logger_mt("davidson");
  if(not print_ci) spdlog::null_logger_mt("ci_solver");
  if(not print_mcscf) spdlog::null_logger_mt("mcscf");
  if(not print_diis) spdlog::null_logger_mt("diis");
  if(not print_asci_search) spdlog::null_logger_mt("asci_search");

  std::vector<double> occs(n_active, 0);
  double dstep = 2.E-2;
  double E = 0.0;
  double nel = 2.0;

  macis::fix_mu_params params;
  params.nbeta = &nbeta;
  params.nalpha = &nalpha;
  params.nel = &nel;
  params.n_active = &n_active;
  params.n_inactive = &n_inactive;
  params.norb = &norb;
  params.n_imp = &n_imp;
  params.E_core = &E_core;
  params.dstep = &dstep;
  params.V = &V;
  params.just_singles = &just_singles;
  params.mcscf_settings = &mcscf_settings;
  params.asci_settings = &asci_settings;
  params.dets = &dets;
  params.C = &C;
  params.occs = &occs;
  params.E = &E;
  params.ci_exp = &ciexp_str;

  // double mu = -9.0;
  // for(int i = 0; i < n_imp; i++)
  // {
  //  T[i*norb+i] = mu;
  // }

  params.T = &T;

  std::cout << "E = " << E << std::endl;

  // for(int i = 0; i < n_active; i++) {
  // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
  // }

  // E = SolveImpurityED(&params);
  // E = SolveImpurityASCI(&params);

  // std::cout << "E = " << E << std::endl;

  // for(int i = 0; i < n_active; i++) {
  // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
  // }

  // Printing Mu_Cost_f and Mu_Cost_df as a function of mu to check if they are
  // correct mu ranging from -10 to 10

  double mu = -10.0;
  double err;
  double derr;
  int npt = 100;
  for(int i = 0; i < npt; i++) {
    mu += 20.0 / npt;
    err = Mu_Cost_f(mu, &params);
    derr = Mu_Cost_df(mu, &params);
    std::cout << mu << " " << err + nel << " " << derr << std::endl;
  }

  // double err = Mu_Cost_f (mu , &params);
  // double derr = Mu_Cost_df (mu , &params);
  // std::cout << "err = " << err << std::endl;
  // std::cout << "derr = " << derr << std::endl;

  // std::cout << "E = " << E << std::endl;

  // for(int i = 0; i < n_active; i++) {
  // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
  // }

  return 0;
}
