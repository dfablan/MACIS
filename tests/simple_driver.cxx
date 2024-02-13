#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <iomanip>
#include <iostream>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/gf/gf.hpp>
// #include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sd_build.hpp>
#include <macis/util/cas.hpp>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/moller_plesset.hpp>
#include <macis/util/mpi.hpp>
#include <macis/util/transform.hpp>
#include <macis/wavefunction_io.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

#include "ini_input.hpp"

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

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
  // XXX: Consider reading this into shared memory to avoid replication
  std::vector<double> T(norb2), V(norb4);
  auto E_core = macis::read_fcidump_core(fcidump_fname);
  macis::read_fcidump_1body(fcidump_fname, T.data(), norb);
  macis::read_fcidump_2body(fcidump_fname, V.data(), norb);
  bool just_singles = macis::is_2body_diagonal(fcidump_fname);
  // std::cout << "just_singles = " << just_singles << "\n" << std::endl;
  // std::cout << "T_diagonal = " << T[1+norb*1] << "\n" << std::endl;
  // std::cout << "T_diagonal = " << T[0+norb*0] << "\n" << std::endl;
  // std::vector<double> * T_ptr = &T;
  // std::cout << "T_diagonal = " << T_ptr->at(1+norb*1) << "\n" << std::endl;

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

  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);
  // Compute active-space Hamiltonian and inactive Fock matrix
  std::vector<double> F_inactive(norb2);
  macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                            NumInactive(n_inactive), T.data(), norb, V.data(),
                            norb, F_inactive.data(), norb, T_active.data(),
                            n_active, V_active.data(), n_active);
  console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(F_inactive));
  console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(V_active));
  console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(T_active));

  // Compute Inactive energy
  auto E_inactive = macis::inactive_energy(NumInactive(n_inactive), T.data(),
                                           norb, F_inactive.data(), norb);
  console->info("E(inactive) = {:.12f}", E_inactive);

  // Storage for active RDMs
  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  double E0 = 0;
  std::cout << "STARTING JOB \n";

  using generator_t = macis::SDBuildHamiltonianGenerator<nwfn_bits>;

  generator_t ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
  ham_gen.SetJustSingles(just_singles);
  ham_gen.SetNimp(n_imp);

  if(ci_exp == CIExpansion::CAS) {
    std::cout << "CIExpansion=CAS \n";
    std::vector<double> C_local;
    // TODO: VERIFY MPI + CAS
    // E0 = macis::CASRDMFunctor<generator_t>::rdms(
    //     mcscf_settings, NumOrbital(n_active), nalpha, nbeta,
    //     T_active.data(), V_active.data(), active_ordm.data(),
    //     active_trdm.data(), C_local MACIS_MPI_CODE(, MPI_COMM_WORLD));

    dets = macis::generate_hilbert_space<nwfn_bits>(norb, nalpha, nbeta);
    E0 = macis::selected_ci_diag(
        dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
        mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
        MACIS_MPI_CODE(MPI_COMM_WORLD, ) true);

    E0 += E_inactive + E_core;

    ham_gen.form_rdms(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C_local.data(),
        macis::matrix_span<double>(active_ordm.data(), norb, norb),
        macis::rank4_span<double>(active_trdm.data(), norb, norb, norb, norb));

    C = C_local;

    if(print_determinants) {
      auto det_logger = spdlog::stdout_color_mt("determinants");
      det_logger->info("Print leading determinants > {:.12f}",
                       determinants_threshold);
      // auto dets = macis::generate_hilbert_space<generator_t::nbits>(
      dets = macis::generate_hilbert_space<generator_t::nbits>(n_active, nalpha,
                                                               nbeta);
      for(size_t i = 0; i < dets.size(); ++i) {
        if(std::abs(C_local[i]) > determinants_threshold) {
          det_logger->info("{:>16.12f}   {}", C_local[i],
                           macis::to_canonical_string(dets[i]));
        }
      }
    }

  } else {
    std::cout << "CIExpansion=ASCI \n";
    // Generate the Hamiltonian Generator
    // generator_t ham_gen(
    //     macis::matrix_span<double>(T_active.data(), n_active, n_active),
    //     macis::rank4_span<double>(V_active.data(), n_active, n_active,
    //                               n_active, n_active));
    if(asci_wfn_fname.size()) {
      // Read wave function from standard file
      console->info("Reading Guess Wavefunction From {}", asci_wfn_fname);
      macis::read_wavefunction(asci_wfn_fname, dets, C);
      // std::cout << dets[0].to_ullong() << std::endl;
      if(compute_asci_E0) {
        console->info("*  Calculating E0");
        E0 = 0;
        for(auto ii = 0; ii < dets.size(); ++ii) {
          double tmp = 0.0;
          for(auto jj = 0; jj < dets.size(); ++jj) {
            tmp += ham_gen.matrix_element(dets[ii], dets[jj]) * C[jj];
          }
          E0 += C[ii] * tmp;
        }
      } else {
        console->info("*  Reading E0");
        E0 = asci_E0 - E_core - E_inactive;
      }
    } else {
      // HF Guess
      console->info("Generating HF Guess for ASCI");
      dets = {macis::canonical_hf_determinant<nwfn_bits>(nalpha, nalpha)};
      // std::cout << dets[0].to_ullong() << std::endl;
      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C = {1.0};
    }
    console->info("ASCI Guess Size = {}", dets.size());
    console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);
    // Perform the ASCI calculation
    auto asci_st = hrt_t::now();
    // Growth phase
    std::cout << "GROWTH PHASE \n";
    std::tie(E0, dets, C) = macis::asci_grow(
        asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
        ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
    // Refinement phase
    std::cout << "REFINEMENT PHASE \n";
    if(asci_settings.max_refine_iter) {
      std::tie(E0, dets, C) = macis::asci_refine(
          asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
          ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
    }
    E0 += E_inactive + E_core;
    auto asci_en = hrt_t::now();
    dur_t asci_dur = asci_en - asci_st;
    console->info("* ASCI_DUR = {:.2e} ms", asci_dur.count());
    if(asci_wfn_out_fname.size()) {
      console->info("Writing ASCI Wavefunction to {}", asci_wfn_out_fname);
      macis::write_wavefunction(asci_wfn_out_fname, n_active, dets, C);
    }

    ham_gen.form_rdms(
        dets.begin(), dets.end(), dets.begin(), dets.end(), C.data(),
        macis::matrix_span<double>(active_ordm.data(), n_active, n_active),
        macis::rank4_span<double>(active_trdm.data(), n_active, n_active,
                                  n_active, n_active));
  }
  console->info("E(CI)  = {:.12f} Eh", E0);

  bool testGF = false;
  OPT_KEYWORD("CI.GF", testGF, bool);
  if(testGF) {
    std::cout << "CI.GF=TRUE \n";
    // if (ci_exp == CIExpansion::CAS) {
    // Generate determinant list
    // auto dets = macis::generate_hilbert_space<generator_t::nbits>(

    // dets = macis::generate_hilbert_space<generator_t::nbits>(
    //     n_active, nalpha, nbeta);
    // }
    // Generate the Hamiltonian Generator
    // generator_t ham_gen(
    //     macis::matrix_span<double>(T_active.data(), n_active, n_active),
    //     macis::rank4_span<double>(V_active.data(), n_active, n_active,
    //                               n_active, n_active));

    // MCSCF Settings
    macis::GFSettings gf_settings;
    OPT_KEYWORD("GF.NORBS", gf_settings.norbs, size_t);
    OPT_KEYWORD("GF.TRUNC_SIZE", gf_settings.trunc_size, size_t);
    OPT_KEYWORD("GF.TOT_SD", gf_settings.tot_SD, int);
    OPT_KEYWORD("GF.GFSEEDTHRES", gf_settings.GFseedThres, double);
    OPT_KEYWORD("GF.ASTHRES", gf_settings.asThres, double);
    OPT_KEYWORD("GF.USE_BANDLAN", gf_settings.use_bandLan, bool);
    OPT_KEYWORD("GF.NLANITS", gf_settings.nLanIts, int);
    OPT_KEYWORD("GF.WRITE", gf_settings.writeGF, bool);
    OPT_KEYWORD("GF.WRITE_SINGLEF", gf_settings.writeGF_singlef, bool);
    OPT_KEYWORD("GF.PRINT", gf_settings.print, bool);
    OPT_KEYWORD("GF.SAVEGFMATS", gf_settings.saveGFmats, bool);
    OPT_KEYWORD("GF.ORBS_BASIS", gf_settings.GF_orbs_basis, std::vector<int>);
    OPT_KEYWORD("GF.IS_UP_BASIS", gf_settings.is_up_basis, std::vector<bool>);
    OPT_KEYWORD("GF.ORBS_COMP", gf_settings.GF_orbs_comp, std::vector<int>);
    OPT_KEYWORD("GF.IS_UP_COMP", gf_settings.is_up_comp, std::vector<bool>);

    // gf_settings.GF_orbs_basis = std::vector<int>(n_active, 0);
    // for(int i = 0; i < n_active; i++) gf_settings.GF_orbs_basis[i] = i;
    // gf_settings.GF_orbs_comp = std::vector<int>(n_active, 0);
    // for(int i = 0; i < n_active; i++) gf_settings.GF_orbs_comp[i] = i;
    // gf_settings.is_up_basis = std::vector<bool>(n_active, true);
    // gf_settings.is_up_comp = std::vector<bool>(n_active, true);

    // Generate frequency grid
    OPT_KEYWORD("GF.WMIN", gf_settings.wmin, double);
    OPT_KEYWORD("GF.WMAX", gf_settings.wmax, double);
    OPT_KEYWORD("GF.NWS", gf_settings.nws, size_t);
    OPT_KEYWORD("GF.ETA", gf_settings.eta, double);
    OPT_KEYWORD("GF.BETA", gf_settings.beta, double);
    OPT_KEYWORD("GF.IMAG_FREQ", gf_settings.imag_freq, bool);
    std::vector<std::complex<double>> ws(gf_settings.nws,
                                         std::complex<double>(0., 0.));

    for(int i = 0; i < gf_settings.nws; i++)
      if(gf_settings.imag_freq) {
        // std::complex<double> w0(0,gf_settings.wmin);
        // std::complex<double> wf(0,gf_settings.wmax);
        // ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);

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
        std::vector<std::complex<double>>(n_active * n_active,
                                          std::complex<double>(0., 0.)));
    std::vector<std::vector<std::complex<double>>> GF_tmp(
        gf_settings.nws,
        std::vector<std::complex<double>>(n_active * n_active,
                                          std::complex<double>(0., 0.)));
    // Occupation numbers
    std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active] / 2;
      std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }
    double curr_nel = std::accumulate(occs.begin(), occs.begin() + n_imp, 0.0);
    std::cout << "Total number of electrons = " << std::setprecision(16)
              << 2 * curr_nel << std::endl;

    // GS vector
    std::vector<int> todelete_p;
    std::vector<int> todelete_h;
    Eigen::VectorXd psi0 =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(C.data(), C.size());

    // Evaluate particle GF
    macis::RunGFCalc<nwfn_bits>(GF_tmp, psi0, ham_gen, dets, E0, true, ws, occs,
                                gf_settings, todelete_p);
    GF = GF_tmp;
    // Evaluate hole GF
    macis::RunGFCalc<nwfn_bits>(GF_tmp, psi0, ham_gen, dets, E0, false, ws,
                                occs, gf_settings, todelete_h);

    if(todelete_h != todelete_p)
      std::cout << "ERROR: todelete_h!=todelete_p" << std::endl;

    GF = macis::sum_GFs(GF, GF_tmp, ws, gf_settings.GF_orbs_comp, todelete_p);

    if(gf_settings.writeGF_singlef)
      macis::write_GF(GF, ws, gf_settings.GF_orbs_comp, todelete_p);
  }

  MACIS_MPI_CODE(MPI_Finalize();)
}
