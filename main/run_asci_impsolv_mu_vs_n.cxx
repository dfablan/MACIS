#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <iomanip>
#include <iostream>
#include <macis/comp_observables.hpp>
#include <macis/doping/fix_mu.hpp>
#include <macis/gf/gf.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

#include "../tests/ini_input.hpp"

constexpr size_t nwfn_bits = 64;

std::map<std::string, CIExpansion> ci_exp_map = {
    {"CAS", CIExpansion::CAS},
    {"ASCI", CIExpansion::ASCI},
    {"ASCI_cheap", CIExpansion::ASCI_cheap}};

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

#ifdef MACIS_ENABLE_MPI
  MACIS_MPI_CODE(MPI_Init(&argc, &argv);)
  auto world_rank = macis::comm_rank(MPI_COMM_WORLD);
  auto world_size = macis::comm_size(MPI_COMM_WORLD);
#else
  int world_rank = 0;
  int world_size = 1;
#endif

  // Create Logger
  auto console = world_rank ? spdlog::null_logger_mt("test_driver_dop")
                            : spdlog::stdout_color_mt("test_driver_dop");

  // Read Input Options
  std::vector<std::string> opts(argc);
  for(int i = 0; i < argc; ++i) opts[i] = argv[i];

  auto input_file = opts.at(1);
  INIFile input(input_file);

  macis::impurity_params<nwfn_bits> params;

  // Required Keywords
  auto fcidump_fname = input.getData<std::string>("CI.FCIDUMP");
  params.nalpha = input.getData<size_t>("CI.NALPHA");
  params.nbeta = input.getData<size_t>("CI.NBETA");

  if(params.nalpha != params.nbeta) throw std::runtime_error("NALPHA != NBETA");

  // Read FCIDUMP File
  params.norb = macis::read_fcidump_norb(fcidump_fname);
  size_t norb2 = params.norb * params.norb;
  size_t norb3 = norb2 * params.norb;
  size_t norb4 = norb2 * norb2;

  // XXX: Consider reading this into shared memory to avoid replication
  params.T.resize(norb2);
  params.V.resize(norb4);
  params.E_core = macis::read_fcidump_core(fcidump_fname);
  macis::read_fcidump_1body(fcidump_fname, params.T.data(), params.norb);
  macis::read_fcidump_2body(fcidump_fname, params.V.data(), params.norb);

#define OPT_KEYWORD(STR, RES, DTYPE) \
  if(input.containsData(STR)) {      \
    RES = input.getData<DTYPE>(STR); \
  }

  // Set up job
  std::string ciexp_str;
  OPT_KEYWORD("CI.EXPANSION", ciexp_str, std::string);
  try {
    params.ci_exp = ci_exp_map.at(ciexp_str);
  } catch(...) {
    throw std::runtime_error("CI Expansion Not Recognized");
  }

  // Set up active space
  params.n_inactive = 0;
  OPT_KEYWORD("CI.NINACTIVE", params.n_inactive, size_t);
  if(params.n_inactive >= params.norb) throw std::runtime_error("NINACTIVE >= NORB");

  params.n_active = params.norb - params.n_inactive;
  OPT_KEYWORD("CI.NACTIVE", params.n_active, size_t);

  if(params.n_inactive + params.n_active > params.norb)
    throw std::runtime_error("NINACTIVE + NACTIVE > NORB");

  size_t n_virtual = params.norb - params.n_active - params.n_inactive;

  params.n_imp = params.norb;
  OPT_KEYWORD("CI.NIMP", params.n_imp, size_t);

  params.nbands = 1;
  OPT_KEYWORD("CI.NBANDS", params.nbands, size_t);
  size_t nsites = params.n_imp / params.nbands;

  // Misc optional files
  std::string rdm_fname, fci_out_fname;
  OPT_KEYWORD("CI.RDMFILE", rdm_fname, std::string);
  OPT_KEYWORD("CI.FCIDUMP_OUT", fci_out_fname, std::string);
  
  bool compute_db_occs = false;
  bool compute_sz_sz = false;
  bool compute_tz_tz = false;
  OPT_KEYWORD("CI.COMP_DB_OCCS", compute_db_occs, bool);
  OPT_KEYWORD("CI.COMP_SZ_I_SZ_J", compute_sz_sz, bool);
  OPT_KEYWORD("CI.COMP_TAUZ_I_TAUZ_J", compute_tz_tz, bool);

  if(params.n_active > nwfn_bits / 2) throw std::runtime_error("Not Enough Bits");

  // MCSCF Settings
  OPT_KEYWORD("MCSCF.MAX_MACRO_ITER", params.mcscf_settings.max_macro_iter, size_t);
  OPT_KEYWORD("MCSCF.MAX_ORB_STEP", params.mcscf_settings.max_orbital_step, double);
  OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL", params.mcscf_settings.orb_grad_tol_mcscf, double);
  OPT_KEYWORD("MCSCF.ENABLE_DIIS", params.mcscf_settings.enable_diis, bool);
  OPT_KEYWORD("MCSCF.DIIS_START_ITER", params.mcscf_settings.diis_start_iter, size_t);
  OPT_KEYWORD("MCSCF.DIIS_NKEEP", params.mcscf_settings.diis_nkeep, size_t);
  OPT_KEYWORD("MCSCF.CI_RES_TOL", params.mcscf_settings.ci_res_tol, double);
  OPT_KEYWORD("MCSCF.CI_MAX_SUB", params.mcscf_settings.ci_max_subspace, size_t);
  OPT_KEYWORD("MCSCF.CI_MATEL_TOL", params.mcscf_settings.ci_matel_tol, double);

  OPT_KEYWORD("MCSCF.CI_NSTATES", params.mcscf_settings.ci_nstates, size_t);

  // ASCI Settings
  std::string asci_wfn_out_fname;
  params.asci_E0 = 0.0;
  params.compute_asci_E0 = true;
  OPT_KEYWORD("ASCI.NTDETS_MAX", params.asci_settings.ntdets_max, size_t);
  OPT_KEYWORD("ASCI.NTDETS_MIN", params.asci_settings.ntdets_min, size_t);
  OPT_KEYWORD("ASCI.NCDETS_MAX", params.asci_settings.ncdets_max, size_t);
  OPT_KEYWORD("ASCI.HAM_EL_TOL", params.asci_settings.h_el_tol, double);
  OPT_KEYWORD("ASCI.RV_PRUNE_TOL", params.asci_settings.rv_prune_tol, double);
  OPT_KEYWORD("ASCI.PAIR_MAX_LIM", params.asci_settings.pair_size_max, size_t);
  OPT_KEYWORD("ASCI.GROW_FACTOR", params.asci_settings.grow_factor, int);
  OPT_KEYWORD("ASCI.MAX_REFINE_ITER", params.asci_settings.max_refine_iter, size_t);
  OPT_KEYWORD("ASCI.REFINE_ETOL", params.asci_settings.refine_energy_tol, double);
  OPT_KEYWORD("ASCI.GROW_WITH_ROT", params.asci_settings.grow_with_rot, bool);
  OPT_KEYWORD("ASCI.GROW_WITH_ROT_LEGACY", params.asci_settings.grow_with_rot_legacy,
              bool);
  OPT_KEYWORD("ASCI.NROTS", params.asci_settings.nrots, size_t);
  OPT_KEYWORD("ASCI.ROT_SIZE_START", params.asci_settings.rot_size_start, size_t);
  OPT_KEYWORD("ASCI.CONSTRAINT_LVL", params.asci_settings.constraint_level, int);
  OPT_KEYWORD("ASCI.WFN_FILE", params.asci_wfn_fname, std::string);
  OPT_KEYWORD("ASCI.WFN_OUT_FILE", asci_wfn_out_fname, std::string);
  if(input.containsData("ASCI.E0_WFN")) {
    params.asci_E0 = input.getData<double>("ASCI.E0_WFN");
    params.compute_asci_E0 = false;
  }

  bool mp2_guess = false;
  OPT_KEYWORD("MCSCF.MP2_GUESS", mp2_guess, bool);

  if(!world_rank) {
    console->info("[Wavefunction Data]:");
    console->info("  * CIEXP   = {}", ciexp_str);
    console->info("  * FCIDUMP = {}", fcidump_fname);
    if(fci_out_fname.size())
      console->info("  * FCIDUMP_OUT = {}", fci_out_fname);
    console->debug("READ {} 1-body integrals and {} 2-body integrals", params.T.size(),
                   params.V.size());
    console->info("ECORE = {:.12f}", params.E_core);
    console->debug("TSUM  = {:.12f}", vec_sum(params.T));
    console->debug("VSUM  = {:.12f}", vec_sum(params.V));
    console->info("TMEM   = {:.2e} GiB", macis::to_gib(params.T));
    console->info("VMEM   = {:.2e} GiB", macis::to_gib(params.V));
  }

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
  spdlog::null_logger_mt("asci_search");

  double nel_target;
  params.occs.resize(params.n_active, 0);
  params.orb_rot.resize(params.n_active * params.n_active);
  for(size_t i = 0; i < params.n_active; ++i) params.orb_rot[i * params.n_active + i] = 1.0;
  params.E = 0.0;
  
  // Copy integrals into active subsets
  params.T_active.resize(params.n_active * params.n_active);
  params.Td_active.resize(params.n_active * params.n_active);
  params.V_active.resize(params.n_active * params.n_active * params.n_active * params.n_active);

  // Compute active-space Hamiltonian and inactive Fock matrix
  params.F_inactive.resize(norb2);
  macis::active_hamiltonian(NumOrbital(params.norb), NumActive(params.n_active),
                            NumInactive(params.n_inactive), params.T.data(), params.norb, params.V.data(),
                            params.norb, params.F_inactive.data(), params.norb, params.T_active.data(),
                            params.n_active, params.V_active.data(), params.n_active);

  console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(params.F_inactive));
  console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(params.V_active));
  console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(params.T_active));

  // Compute Inactive energy
  params.E_inactive = macis::inactive_energy(NumInactive(params.n_inactive), params.T.data(),
                                           params.norb, params.F_inactive.data(), params.norb);
  console->info("E(inactive) = {:.12f}", params.E_inactive);

  bool doping = false;
  OPT_KEYWORD("CI.DOPING", doping, bool);

  OPT_KEYWORD("DOP.NELECTRONS", nel_target, double);

  params.dstep = 2.E-2;
  params.abs_tol = 1.E-4;
  params.maxiter = 100;
  params.print_doping = true;
  double init_shift = 2.0;
  bool deriv;
  std::string method_name;

  OPT_KEYWORD("DOP.DERIV", deriv, bool);
  OPT_KEYWORD("DOP.ABS_TOL", params.abs_tol, double);
  OPT_KEYWORD("DOP.MAXITER", params.maxiter, size_t);
  OPT_KEYWORD("DOP.PRINT_DOPING", params.print_doping, bool);
  OPT_KEYWORD("DOP.INIT_SHIFT", params.init_shift, double);
  OPT_KEYWORD("DOP.DSTEP", params.dstep, double);
  OPT_KEYWORD("DOP.METHOD", method_name, std::string);
  params.delta_CFS = 0.0;
  OPT_KEYWORD("DOP.DELTA_CFS", params.delta_CFS, double);

  std::cout << "Electron filling parameters \n";
  std::cout << std::setprecision(3) << nel_target
            << " electrons per orbital \n";
  std::cout << std::setprecision(2) << nel_target * params.n_imp << " electrons in "
            << std::setprecision(1) << params.n_imp << " orbitals \n";

  //Doping Parameters
  double init_mu;
  OPT_KEYWORD("DOP.INIT_MU", init_mu, double);
  double final_mu;
  OPT_KEYWORD("DOP.FINAL_MU", final_mu, double);
  size_t mu_npt; 
  OPT_KEYWORD("DOP.MU_NPT", mu_npt, size_t);
 

  std::cout << "List of mu values that will be used to compute n(mu):" << std::endl;
  for (int i = 0; i < mu_npt; i++)
  {
    double x = init_mu + i*(final_mu - init_mu)/(mu_npt-1);
    std::cout << std::setprecision(10) << x << ", ";
  }
  std::cout << "\n--------------------------" << std::endl;
  std::ofstream ofile( "mu_vs_n.dat", std::ios::out ); 
  ofile.precision(std::numeric_limits<double>::max_digits10);
  ofile << "# mu  n" << std::endl;
  for (int i = 0; i < mu_npt; i++)
  {
    double mu = init_mu + i*(final_mu - init_mu)/(mu_npt-1);
    double n = macis::Mu_vs_n<nwfn_bits>(mu, &params);
    ofile << mu << "  " << n << std::endl;
  }





  return 0;
}
