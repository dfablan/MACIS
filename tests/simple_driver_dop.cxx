#include <macis/doping/call_solver.hpp>
#include <macis/doping/fix_mu.hpp>

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <iomanip>
#include <iostream>

#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

#include "ini_input.hpp"
#include <macis/gf/gf.hpp>




template <typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

enum class CIExpansion {CAS , ASCI};

std::map<std::string, CIExpansion> ci_exp_map = {{"CAS", CIExpansion::CAS},
                                                 {"ASCI", CIExpansion::ASCI}};

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


  std::string ciexp_str ;
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
  OPT_KEYWORD("MCSCF.MCSCF_ORB_TOL", mcscf_settings.orb_grad_tol_mcscf,
              double);
  // OPT_KEYWORD("MCSCF.BFGS_TOL",       mcscf_settings.orb_grad_tol_bfgs,
  // double); OPT_KEYWORD("MCSCF.BFGS_MAX_ITER", mcscf_settings.max_bfgs_iter,
  // size_t);
  OPT_KEYWORD("MCSCF.ENABLE_DIIS", mcscf_settings.enable_diis, bool);
  OPT_KEYWORD("MCSCF.DIIS_START_ITER", mcscf_settings.diis_start_iter,
              size_t);
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
  if(fci_out_fname.size())
    console->info("  * FCIDUMP_OUT = {}", fci_out_fname);
  console->debug("READ {} 1-body integrals and {} 2-body integrals",
                 T.size(), V.size());
  console->info("ECORE = {:.12f}", E_core);
  console->debug("TSUM  = {:.12f}", vec_sum(T));
  console->debug("VSUM  = {:.12f}", vec_sum(V));
  console->info("TMEM   = {:.2e} GiB", macis::to_gib(T));
  console->info("VMEM   = {:.2e} GiB", macis::to_gib(V));
 
  // Setup printing
  bool print_davidson = true, print_ci = true, print_mcscf = true,
       print_diis = true, print_asci_search = true,
       print_determinants = true;
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
  if(not print_asci_search)
  spdlog::null_logger_mt("asci_search");

  double nel ;
  std::vector<double> occs(n_active, 0);
  double E0=0.0;
  // bool asci_wfn_guess = false;

  macis::fix_mu_params params;
  params.nbeta = &nbeta;
  params.nalpha = &nalpha;
  params.n_active = &n_active;
  params.n_inactive = &n_inactive;
  params.norb = &norb;
  params.n_imp = &n_imp;
  params.E_core = &E_core;
  params.V = &V;
  params.T = &T;
  params.just_singles = &just_singles;
  params.mcscf_settings = &mcscf_settings;
  params.asci_settings = &asci_settings;
  params.dets = &dets;
  params.C = &C;
  params.occs = &occs;
  params.E = &E0;
  params.asci_wfn_fname = &asci_wfn_fname;
  params.compute_asci_E0 = &compute_asci_E0;
  params.asci_E0 = &asci_E0;




  bool doping = false;
  OPT_KEYWORD("CI.DOPING",doping, bool);

  OPT_KEYWORD("DOP.NELECTRONS",nel, double);

  if(doping && nel/n_imp==1)
    std::cout << "WARNING: Doping routines were called but half-filling was asked \n" ;
    std::cout << "Doping =" << doping << std::endl ;
  if(doping)
  {
    double dstep = 2.E-2;
    double init_mu = -9.5;
    double abs_tol =  1.E-4; 
    size_t maxiter = 100; 
    bool print_doping =  true; 
    double init_shift =  2.0;
    bool deriv;
    std::string method_name;

    OPT_KEYWORD("DOP.DERIV",deriv, bool);
    OPT_KEYWORD("DOP.INIT_MU",init_mu, double);
    OPT_KEYWORD("DOP.ABS_TOL",abs_tol, double);
    OPT_KEYWORD("DOP.MAXITER",maxiter, size_t);
    OPT_KEYWORD("DOP.PRINT_DOPING",print_doping, bool);
    OPT_KEYWORD("DOP.INIT_SHIFT",init_shift, double);
    OPT_KEYWORD("DOP.DSTEP",dstep, double);    
    OPT_KEYWORD("DOP.METHOD",method_name, std::string);

    std::cout << "Electron filling parameters \n" ;
    std::cout << std::setprecision(2) << nel<< " electrons in " << std::setprecision(1) << n_imp <<  " orbitals \n" ;
    std::cout << std::setprecision(3) << nel/n_imp << " electrons per orbital \n" ;

    params.dstep = &dstep;
    params.abs_tol = &abs_tol;
    params.maxiter = &maxiter;
    params.print = &print_doping;
    params.init_shift = &init_shift;
    params.ci_exp = &ciexp_str;
    params.nel = &nel;

    double mu_fixed;

    if(deriv)
      mu_fixed = Fix_Mu_der(method_name, init_mu, &params);
    else
      mu_fixed = Fix_Mu_noder(method_name, init_mu, &params);

    std::cout << "Mu has been fixed to " << std::setprecision(10) << mu_fixed << std::endl;
    
    // std::cout << "The current occupation values are: \n";
    // for(int i = 0; i < n_active; i++) {
    //   std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    // }

    // std::cout << "The current GS energy is: \n";
    // std::cout << "E = " << E << std::endl;
  
    double curr_nel = std::accumulate(occs.begin(), occs.begin()+n_imp, 0.0);
    std::cout<< "Total number of electrons = "<< std::setprecision(10) << curr_nel << std::endl;

    // Write new FCIDUMP file for the impurity orbitals
    std::string fcilocal_out_fname = "locFCIDUMP.dat";
    macis::write_fcidump(fcilocal_out_fname, n_imp,T.data(), norb, V.data(), norb,
                     E_core);

    if(ci_exp == CIExpansion::ASCI && asci_wfn_out_fname.size()) 
    {
        console->info("Writing ASCI Wavefunction to {}", asci_wfn_out_fname);
       macis::write_wavefunction(asci_wfn_out_fname, n_active, dets, C);
    }

  }
  
  else
  {

    std::cout << "Doping routines have not been called\n" ;
    std::cout << "mu should be equal to -U/2 for have filling in single band models\n" ;

    if(ci_exp == CIExpansion::CAS) 
    { 
        E0 = SolveImpurityED(&params);

        if(print_determinants) 
        {
          auto det_logger = spdlog::stdout_color_mt("determinants");
          det_logger->info("Print leading determinants > {:.12f}",
                              determinants_threshold);
          // auto dets = macis::generate_hilbert_space<generator_t::nbits>(
          // dets = macis::generate_hilbert_space<generator_t::nbits>(
          //     n_active, nalpha, nbeta);
          for(size_t i = 0; i < dets.size(); ++i) {
            if(std::abs(C[i]) > determinants_threshold) {
              det_logger->info("{:>16.12f}   {}", C[i],
                               macis::to_canonical_string(dets[i]));
            }
           }
        }
    }
    else 
    {
        E0 = SolveImpurityASCI(&params);

        if(asci_wfn_out_fname.size()) 
        {
            console->info("Writing ASCI Wavefunction to {}", asci_wfn_out_fname);
           macis::write_wavefunction(asci_wfn_out_fname, n_active, dets, C);
        }

    }

  }

  console->info("E(CI)  = {:.12f} Eh", E0);
  double curr_nel = std::accumulate(occs.begin(), occs.begin()+n_imp, 0.0);
  std::cout<< "Total number of electrons = "<< curr_nel << std::endl;

  bool testGF = false;
  OPT_KEYWORD("CI.GF", testGF, bool);
  if(testGF) 
  {

    std::cout << "CI.GF=TRUE \n";
    // if (ci_exp == CIExpansion::CAS) {
    // Generate determinant list
    // auto dets = macis::generate_hilbert_space<generator_t::nbits>(
    
    // dets = macis::generate_hilbert_space<generator_t::nbits>(
    //     n_active, nalpha, nbeta);
    // }
  
    // Copy integrals into active subsets
    std::vector<double> T_active(n_active * n_active);
    std::vector<double> V_active(n_active * n_active * n_active * n_active) ;
    // Compute active-space Hamiltonian and inactive Fock matrix
    std::vector<double> F_inactive(norb2);
    macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active),
                             NumInactive(n_inactive), T.data(), norb, V.data(),
                             norb, F_inactive.data(), norb, T_active.data(),
                             n_active, V_active.data(), n_active) ;
    
    // Generate the Hamiltonian Generator
    macis::SDBuildHamiltonianGenerator<nwfn_bits> ham_gen(
      macis::matrix_span<double>(T_active.data(), n_active, n_active),
      macis::rank4_span<double>(V_active.data(), n_active, n_active,
                                n_active, n_active));

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
    OPT_KEYWORD("GF.ORBS_BASIS", gf_settings.GF_orbs_basis,
                std::vector<int>);
    OPT_KEYWORD("GF.IS_UP_BASIS", gf_settings.is_up_basis,
                std::vector<bool>);
    OPT_KEYWORD("GF.ORBS_COMP", gf_settings.GF_orbs_comp,
                std::vector<int>);
    OPT_KEYWORD("GF.IS_UP_COMP", gf_settings.is_up_comp,
                std::vector<bool>);

    //gf_settings.GF_orbs_basis = std::vector<int>(n_active, 0);
    //for(int i = 0; i < n_active; i++) gf_settings.GF_orbs_basis[i] = i;
    //gf_settings.GF_orbs_comp = std::vector<int>(n_active, 0);
    //for(int i = 0; i < n_active; i++) gf_settings.GF_orbs_comp[i] = i;
    //gf_settings.is_up_basis = std::vector<bool>(n_active, true);
    //gf_settings.is_up_comp = std::vector<bool>(n_active, true);

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
      if (gf_settings.imag_freq) 
      {
 
 
        // std::complex<double> w0(0,gf_settings.wmin);
        // std::complex<double> wf(0,gf_settings.wmax);
        // ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);
 
        //  MATSUBARA GRID
        ws[i] = std::complex<double>(0.,(2*i+1)*M_PI/ gf_settings.beta);

      }
      else
      {
        std::complex<double> w0(gf_settings.wmin, gf_settings.eta);
        std::complex<double> wf(gf_settings.wmax, gf_settings.eta);
        ws[i] = w0 + (wf - w0) / double(gf_settings.nws - 1) * double(i);
      }

    // GF vector
    std::vector<std::vector<std::complex<double>>> GF(
        gf_settings.nws, std::vector<std::complex<double>>(
                 n_active*n_active, std::complex<double>(0., 0.)));
    std::vector<std::vector<std::complex<double>>> GF_tmp(
        gf_settings.nws, std::vector<std::complex<double>>(
                 n_active*n_active, std::complex<double>(0., 0.)));
    
    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) 
    {
      // occs[i] = active_ordm[i + i * n_active]/2; 
      occs[i] = occs[i]/2; 
      std::cout << "occs[" << i << "] = " << std::setprecision(10)<< occs[i] << std::endl;
    }
  
    // GS vector
    std::vector<int> todelete_p; 
    std::vector<int> todelete_h; 
    Eigen::VectorXd psi0 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        C.data(), C.size());

    // Evaluate particle GF
    macis::RunGFCalc<nwfn_bits>(GF_tmp, psi0, ham_gen, dets, E0, true, ws,
                              occs, gf_settings,todelete_p);
    GF=GF_tmp;
 
    // Evaluate hole GF
    macis::RunGFCalc<nwfn_bits>(GF_tmp, psi0, ham_gen, dets, E0, false, ws,
                              occs, gf_settings, todelete_h);


    if (todelete_h!=todelete_p)
      std::cout << "ERROR: todelete_h!=todelete_p" << std::endl;

    GF=macis::sum_GFs(GF,GF_tmp,ws,gf_settings.GF_orbs_comp,todelete_p);

    if(gf_settings.writeGF_singlef) macis::write_GF(GF, ws, gf_settings.GF_orbs_comp,todelete_p);

  }

  return 0;
}
