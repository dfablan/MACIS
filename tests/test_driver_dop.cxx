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

// enum class CIExpansion { CAS, ASCI, ASCI_cheap };
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

  // Possibility of hoppings for the spin-down orbitals
  std::string fcidump_do_fname = "NONE";
  params.Td.resize(norb2);
  params.spin_dep = false;
  OPT_KEYWORD("CI.FCIDUMP_DO", fcidump_do_fname, std::string);
  if(fcidump_do_fname != "NONE") {
    macis::read_fcidump_1body(fcidump_do_fname, params.Td.data(), norb);
  params.spin_dep = true;
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
  if(params.n_inactive >= params.norb)
    throw std::runtime_error("NINACTIVE >= NORB");

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

  if(params.n_active > nwfn_bits / 2)
    throw std::runtime_error("Not Enough Bits");

  // MCSCF Settings
  OPT_KEYWORD("MCSCF.MAX_MACRO_ITER", params.mcscf_settings.max_macro_iter,
              size_t);
  OPT_KEYWORD("MCSCF.MAX_ORB_STEP", params.mcscf_settings.max_orbital_step,
              double);
  OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL", params.mcscf_settings.orb_grad_tol_mcscf,
              double);
  OPT_KEYWORD("MCSCF.ENABLE_DIIS", params.mcscf_settings.enable_diis, bool);
  OPT_KEYWORD("MCSCF.DIIS_START_ITER", params.mcscf_settings.diis_start_iter,
              size_t);
  OPT_KEYWORD("MCSCF.DIIS_NKEEP", params.mcscf_settings.diis_nkeep, size_t);
  OPT_KEYWORD("MCSCF.CI_RES_TOL", params.mcscf_settings.ci_res_tol, double);
  OPT_KEYWORD("MCSCF.CI_MAX_SUB", params.mcscf_settings.ci_max_subspace,
              size_t);
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
  OPT_KEYWORD("ASCI.MAX_REFINE_ITER", params.asci_settings.max_refine_iter,
              size_t);
  OPT_KEYWORD("ASCI.REFINE_ETOL", params.asci_settings.refine_energy_tol,
              double);
  OPT_KEYWORD("ASCI.GROW_WITH_ROT", params.asci_settings.grow_with_rot, bool);
  OPT_KEYWORD("ASCI.GROW_WITH_ROT_LEGACY",
              params.asci_settings.grow_with_rot_legacy, bool);
  OPT_KEYWORD("ASCI.NROTS", params.asci_settings.nrots, size_t);
  OPT_KEYWORD("ASCI.ROT_SIZE_START", params.asci_settings.rot_size_start,
              size_t);
  OPT_KEYWORD("ASCI.CONSTRAINT_LVL", params.asci_settings.constraint_level,
              int);
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
    console->debug("READ {} 1-body integrals and {} 2-body integrals",
                   params.T.size(), params.V.size());
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

  params.occs.resize(params.n_active, 0);
  params.orb_rot.resize(params.n_active * params.n_active);
  for(size_t i = 0; i < params.n_active; ++i)
    params.orb_rot[i * params.n_active + i] = 1.0;
  params.E = 0.0;

  // Copy integrals into active subsets
  params.T_active.resize(params.n_active * params.n_active);
  params.Td_active.resize(params.n_active * params.n_active);
  params.V_active.resize(params.n_active * params.n_active * params.n_active *
                         params.n_active);

  // Compute active-space Hamiltonian and inactive Fock matrix
  params.F_inactive.resize(norb2);
  params.Fd_inactive.resize(norb2);
  macis::active_hamiltonian(NumOrbital(params.norb), NumActive(params.n_active),
                            NumInactive(params.n_inactive), params.T.data(),
                            params.norb, params.V.data(), params.norb,
                            params.F_inactive.data(), params.norb,
                            params.T_active.data(), params.n_active,
                            params.V_active.data(), params.n_active);
  if(params.spin_dep)
    macis::active_hamiltonian(
        NumOrbital(params.norb), NumActive(params.n_active), NumInactive(params.n_inactive),
        params.Td.data(), params.norb, params.V.data(), params.norb, params.Fd_inactive.data(), params.norb,
        params.Td_active.data(), params.n_active, params.V_active.data(), params.n_active);

  console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(params.F_inactive));
  console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(params.V_active));
  console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(params.T_active));

  // Compute Inactive energy
  params.E_inactive = macis::inactive_energy(
      NumInactive(params.n_inactive), params.T.data(), params.norb,
      params.F_inactive.data(), params.norb);
  if(params.spin_dep) {
    for(int ii = 0; ii < params.n_inactive; ii++)
      params.E_inactive += params.Td[ii * (1 + params.n_inactive)] - params.T[ii * (1 + params.n_inactive)];
  }
  console->info("E(inactive) = {:.12f}", params.E_inactive);

  bool doping = false;
  OPT_KEYWORD("CI.DOPING", doping, bool);

  OPT_KEYWORD("DOP.NELECTRONS", params.nel_target, double);

  if(doping && params.nel_target / params.n_imp == 1)
    std::cout
        << "WARNING: Doping routines were called but half-filling was asked \n";
  std::cout << "Doping =" << doping << std::endl;

  double E0 = 0.0;

  if(doping) {
    double init_mu = -9.5;
    params.dstep = 2.E-2;
    params.abs_tol = 1.E-4;
    params.maxiter = 100;
    params.print_doping = true;
    params.init_shift = 2.0;
    bool deriv = false;
    std::string method_name = "";

    OPT_KEYWORD("DOP.INIT_MU", init_mu, double);
    OPT_KEYWORD("DOP.DERIV", deriv, bool);
    OPT_KEYWORD("DOP.ABS_TOL", params.abs_tol, double);
    OPT_KEYWORD("DOP.MAXITER", params.maxiter, size_t);
    OPT_KEYWORD("DOP.PRINT_DOPING", params.print_doping, bool);
    OPT_KEYWORD("DOP.INIT_SHIFT", params.init_shift, double);
    OPT_KEYWORD("DOP.DSTEP", params.dstep, double);
    OPT_KEYWORD("DOP.METHOD", method_name, std::string);
    params.delta_CFS = 0.0;
    OPT_KEYWORD("DOP.DELTA_CFS", params.delta_CFS, double);
    params.cheap_mode = false;
    OPT_KEYWORD("DOP.CHEAP_MODE", params.cheap_mode, bool);

    std::cout << "Electron filling parameters \n";
    std::cout << std::setprecision(3) << params.nel_target
              << " electrons per orbital \n";
    std::cout << std::setprecision(2) << params.nel_target * params.n_imp
              << " electrons in " << std::setprecision(1) << params.n_imp
              << " orbitals \n";

    double mu_fixed;

    if(deriv)
      mu_fixed = macis::Fix_Mu_der<nwfn_bits>(method_name, init_mu, &params);
    else
      mu_fixed = macis::Fix_Mu_noder<nwfn_bits>(method_name, init_mu, &params);

    std::cout << "Mu has been fixed to " << std::setprecision(10) << mu_fixed
              << std::endl;

    // std::cout << "The current occupation values are: \n";
    // for(int i = 0; i < n_active; i++) {
    //   std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    // }

    // std::cout << "The current GS energy is: \n";
    // std::cout << "E = " << E << std::endl;

    std::cout << "\nOrbital Occupations (per spin) in the original basis: "
              << std::endl;
    std::cout << "Occs: ";
    for(const auto oc : params.occs) std::cout << oc << ", ";
    std::cout << std::endl;

    double curr_nel =
        2 * std::accumulate(params.occs.begin(),
                            params.occs.begin() + params.n_imp, 0.0);
    std::cout << "Total number of electrons = " << curr_nel << " in "
              << params.n_imp << " impurity orbitals\n"
              << std::endl;

    // Write new FCIDUMP file for the impurity orbitals
    std::string fcilocal_out_fname = "locFCIDUMP.dat";
    macis::write_fcidump(fcilocal_out_fname, params.n_imp, params.T.data(),
                         params.norb, params.V.data(), params.norb,
                         params.E_core);

    if(params.ci_exp == CIExpansion::ASCI && asci_wfn_out_fname.size()) {
      console->info("Writing ASCI Wavefunction to {}", asci_wfn_out_fname);
      macis::write_wavefunction(asci_wfn_out_fname, params.n_active,
                                params.dets, params.C);
    }

    E0 = params.E;
  }

  else {
    std::cout << "Doping routines have not been called\n";
    std::cout << "mu should be equal to -U/2 for have filling in single band "
                 "models\n";
    if(params.ci_exp == CIExpansion::CAS) {
      E0 = macis::SolveImpurityED<nwfn_bits>(&params);
    } else if(params.ci_exp == CIExpansion::ASCI_cheap) {
      E0 = macis::SolveImpurityCheapASCI<nwfn_bits>(&params);
    } else if(params.ci_exp == CIExpansion::ASCI) {
      E0 = macis::SolveImpurityASCI_rot<nwfn_bits>(&params);
      if(asci_wfn_out_fname.size()) {
        console->info("Writing ASCI Wavefunction to {}", asci_wfn_out_fname);
        macis::write_wavefunction(asci_wfn_out_fname, params.n_active,
                                  params.dets, params.C);
      }
    }
  }

  console->info("E(CI)  = {:.12f} Eh", E0);

  std::cout << "\nOrbital Occupations (per spin) in the original basis: "
            << std::endl;
  std::cout << "Occs: ";
  for(const auto oc : params.occs) std::cout << oc << ", ";
  std::cout << std::endl;

  double curr_nel =
      2 * std::accumulate(params.occs.begin(),
                          params.occs.begin() + params.n_imp, 0.0);
  std::cout << "Total number of electrons = " << curr_nel << " in "
            << params.n_imp << " impurity orbitals\n"
            << std::endl;

  if(compute_db_occs or compute_sz_sz or compute_tz_tz) {
    using dbl = std::numeric_limits<double>;
    macis::CompObservables<nwfn_bits> obs(&params);
    if(compute_db_occs) {
      double db_occs = obs.compute_double_occupancies();
      std::cout << "  * Double occupancy = " << db_occs << std::endl;
    }
    if(compute_sz_sz) {
      std::cout << "  * Computing <Sz(i) Sz(j)> correlations" << std::endl;
      std::vector<double> sz_sz(nsites * nsites, 0.0);
      sz_sz = obs.compute_sz_sz_correlations();
      macis::util::write_matrix(sz_sz.data(), nsites, nsites, "sz_sz.dat",
                                true);
    }
    if(compute_tz_tz) {
      std::vector<double> tz_tz(nsites * nsites, 0.0);
      tz_tz = obs.compute_tz_tz_correlations();
      // print to file
      macis::util::write_matrix(tz_tz.data(), nsites, nsites, "tauz_tauz.dat",
                                true);
    }
  }

  bool testGF = false;
  OPT_KEYWORD("CI.GF", testGF, bool);
  if(testGF) {
    params.T_active.assign(params.T_active.size(), 0.0);
    params.V_active.assign(params.V_active.size(), 0.0);
    params.F_inactive.assign(params.F_inactive.size(), 0.0);
    // Compute active-space Hamiltonian and inactive Fock matrix
    macis::active_hamiltonian(
        NumOrbital(params.norb), NumActive(params.n_active),
        NumInactive(params.n_inactive), params.T.data(), params.norb,
        params.V.data(), params.norb, params.F_inactive.data(), params.norb,
        params.T_active.data(), params.n_active, params.V_active.data(),
        params.n_active);

    // Generate the Hamiltonian Generator
    macis::SDBuildHamiltonianGenerator<nwfn_bits> ham_gen(
        macis::matrix_span<double>(params.T_active.data(), params.n_active,
                                   params.n_active),
        macis::rank4_span<double>(params.V_active.data(), params.n_active,
                                  params.n_active, params.n_active,
                                  params.n_active));

    // MCSCF Settings
    macis::GFSettings gf_settings;
    OPT_KEYWORD("GF.NORBS", gf_settings.norbs, size_t);
    OPT_KEYWORD("GF.TRUNC_SIZE", gf_settings.trunc_size, size_t);
    OPT_KEYWORD("GF.TOT_SD", gf_settings.tot_SD, int);
    OPT_KEYWORD("GF.GFSEEDTHRES", gf_settings.GFseedThres, double);
    OPT_KEYWORD("GF.ASTHRES", gf_settings.asThres, double);
    OPT_KEYWORD("GF.USE_BANDLAN", gf_settings.use_bandLan, bool);
    OPT_KEYWORD("GF.NLANITS", gf_settings.nLanIts, int);
    OPT_KEYWORD("GF.WRITE", gf_settings.writeGF_singlef, bool);
    OPT_KEYWORD("GF.PRINT", gf_settings.print, bool);
    OPT_KEYWORD("GF.SAVEGFMATS", gf_settings.saveGFmats, bool);
    OPT_KEYWORD("GF.ORBS_BASIS", gf_settings.GF_orbs_basis, std::vector<int>);
    OPT_KEYWORD("GF.IS_UP_BASIS", gf_settings.is_up_basis, std::vector<bool>);
    OPT_KEYWORD("GF.ORBS_COMP", gf_settings.GF_orbs_comp, std::vector<int>);
    OPT_KEYWORD("GF.IS_UP_COMP", gf_settings.is_up_comp, std::vector<bool>);

    // Generate frequency grid
    OPT_KEYWORD("GF.WMIN", gf_settings.wmin, double);
    OPT_KEYWORD("GF.WMAX", gf_settings.wmax, double);
    OPT_KEYWORD("GF.NWS", gf_settings.nws, size_t);
    OPT_KEYWORD("GF.ETA", gf_settings.eta, double);
    OPT_KEYWORD("GF.BETA", gf_settings.beta, double);
    bool imag_freq = true;
    OPT_KEYWORD("GF.IMAG_FREQ", imag_freq, bool);
    std::vector<std::complex<double>> ws(gf_settings.nws,
                                         std::complex<double>(0., 0.));

    for(int i = 0; i < gf_settings.nws; i++)
      if(imag_freq) {
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
        std::vector<std::complex<double>>(params.n_active * params.n_active,
                                          std::complex<double>(0., 0.)));
    std::vector<std::vector<std::complex<double>>> GF_tmp(
        gf_settings.nws,
        std::vector<std::complex<double>>(params.n_active * params.n_active,
                                          std::complex<double>(0., 0.)));

    // Occupation numbers
    // for(int i = 0; i < n_active; i++)
    // {
    // occs[i] = active_ordm[i + i * n_active]/2;
    // occs[i] = occs[i]/2;
    // std::cout << "occs[" << i << "] = " << std::setprecision(10)<< occs[i] <<
    // std::endl;
    // }

    // GS vector
    std::vector<int> todelete_p;
    std::vector<int> todelete_h;
    Eigen::VectorXd psi0 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        params.C.data(), params.C.size());

    // Evaluate particle GF
    macis::RunGFCalc<nwfn_bits>(GF_tmp, psi0, ham_gen, params.dets, E0, true,
                                ws, params.occs, gf_settings);
    GF = GF_tmp;

    // Evaluate hole GF
    macis::RunGFCalc<nwfn_bits>(GF_tmp, psi0, ham_gen, params.dets, E0, false,
                                ws, params.occs, gf_settings);

    if(todelete_h != todelete_p)
      std::cout << "ERROR: todelete_h!=todelete_p" << std::endl;

    GF = macis::sum_GFs(GF, GF_tmp, ws, gf_settings.GF_orbs_comp, todelete_p);

    if(gf_settings.writeGF_singlef)
      macis::write_GF(GF, ws, gf_settings.GF_orbs_comp, todelete_p);
  }

  return 0;
}
