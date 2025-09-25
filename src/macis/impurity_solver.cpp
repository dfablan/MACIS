#define IMPURITY_SOLVER_CPP
#include "macis/impurity_solver.hpp"


namespace macis {

  template <size_t N>
  auto evaluate_GF(
  const double EASCI,
  macis::impurity_params<N>& p,
  const macis::DoubleLoopHamiltonianGenerator<N> &ham_gen,
  macis::GFSettings &gf_settings
) {
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
    std::vector<std::vector<std::complex<double>>> GF( gf_settings.nws,
        std::vector<std::complex<double>>(p.n_active * p.n_active,
                                          std::complex<double>(0., 0.)));
    std::vector<std::vector<std::complex<double>>> GF_tmp( gf_settings.nws,
        std::vector<std::complex<double>>(p.n_active * p.n_active,
                                          std::complex<double>(0., 0.)));

    // GS vector
    std::vector<int> todelete_p;
    std::vector<int> todelete_h;
    Eigen::VectorXd psi0 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        p.C.data(), p.C.size());

    // Evaluate particle GF
    macis::RunGFCalc<N>(GF_tmp, psi0, ham_gen, p.dets, EASCI, true,
                                ws, p.occs, gf_settings);
    GF = GF_tmp;

    // Evaluate hole GF
    macis::RunGFCalc<N>(GF_tmp, psi0, ham_gen, p.dets, EASCI, false,
                                ws, p.occs, gf_settings);

    if(todelete_h != todelete_p)
      std::cout << "ERROR: todelete_h!=todelete_p" << std::endl;

    GF = macis::sum_GFs(GF, GF_tmp, ws, gf_settings.GF_orbs_comp, todelete_p);

    // Rotate the GF back to original basis
    // for( int iw = 0; iw < GF.size(); iw++)
    // {
    //  Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero( GF[0].size(), GF[0][0].size());
    //  for( int j = 0; j < GF[iw].size(); j++)
    //    for( int k = 0; k < GF[iw][j].size(); k++)
    //      G(j, k) = GF[iw][j][k];
    //  Eigen::MatrixXd rotMat = Eigen::MatrixXd::Identity( GF[0].size(), GF[0][0].size() );
    //  rotMat.block(0,0,imp_rot.rows(), imp_rot.cols()) = imp_rot; 
    //  Eigen::MatrixXcd rotG  = rotMat.adjoint() * G * rotMat;
    //  for( int j = 0; j < GF[iw].size(); j++)
    //    for( int k = 0; k < GF[iw][j].size(); k++)
    //      GF[iw][j][k] = rotG(j, k);
    // }

    if(gf_settings.writeGF_singlef)
      macis::write_GF(GF, ws, gf_settings.GF_orbs_comp, todelete_p);


    return GF;
}


template <size_t N>
auto evaluate_ordm(
  std::vector<macis::wfn_t<N>> &dets,
  std::vector<double> &X_local,
  macis::DoubleLoopHamiltonianGenerator<N> &ham_gen,
  std::vector<double> &orb_rot
) {
  // Get Parameters
  size_t n_active = sqrt(orb_rot.size());

  // Compute the 1-rdm
  typename std::vector<macis::wfn_t<N>>::iterator det_st = dets.begin();
  typename std::vector<macis::wfn_t<N>>::iterator det_en = dets.end();

  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), X_local.data(), 
  macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
  macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

  // {
  //   //print ordm DEBUG
      //  macis::util::write_matrix(active_ordm.data(), n_active, n_active,
      //                            "active_ordm_rotated.dat", true);

  // Rotate the 1-RDM back to original basis
  // Eigen::MatrixXd roto = orb_rot * o * orb_rot.adjoint();
  std::vector<double> tmp (n_active*n_active, 0. );
  std::vector<double> comp( n_active * n_active, 0. );
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
            n_active, n_active, n_active, 1.0, active_ordm.data(), n_active,
            orb_rot.data(), n_active, 0.0, tmp.data(), n_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            n_active, n_active, n_active, 1.0, orb_rot.data(), n_active,
            tmp.data(), n_active, 0.0, comp.data(), n_active);

  return comp;
}
  
template <size_t N>
double SolveImpurityED (impurity_params<N>& p){

    size_t& norb =  p.norb;
    double& asci_E0 = p.asci_E0;
    size_t& n_active =  p.n_active;
    size_t& nalpha =  p.nalpha;
    size_t& nbeta =  p.nbeta;
    size_t& n_inactive =  p.n_inactive;
    std::vector<double>& T =  p.T;
    std::vector<double>& V =  p.V;
    size_t& n_imp =  p.n_imp;
    double& E_core =  p.E_core;
    macis::MCSCFSettings& mcscf_settings =  p.mcscf_settings;
    macis::ASCISettings& asci_settings =  p.asci_settings;

    // std::vector<macis::wfn_t<N>> dets;
    // std::vector<double> C_local;  
    // std::vector<double> occs(n_active, 0);

    std::vector<macis::wfn_t<N>>& dets = p.dets;
    dets.clear();
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    std::cout << "-------------------------------Entering SolverImpurityED-------------------------------" << std::endl;


    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    double E0 = 0 ;

    using generator_t = macis::DoubleLoopHamiltonianGenerator<N>;

    E0 = macis::CASRDMFunctor<generator_t>::rdms(
          mcscf_settings, NumOrbital(n_active), nalpha, nbeta,
          T_active.data(), V_active.data(), active_ordm.data(),
          active_trdm.data(), C_local MACIS_MPI_CODE(, MPI_COMM_WORLD));
    E0 += E_inactive + E_core;
    
    dets = macis::generate_hilbert_space<generator_t::nbits>(
        n_active, nalpha, nbeta);
    
    // Occupation numbers
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2; 
    }

    return E0;
}

template <size_t N>
double SolveImpurityASCI (impurity_params<N>& p){

    bool& compute_asci_E0 = p.compute_asci_E0;
    double& asci_E0 = p.asci_E0;
    std::string& asci_wfn_fname = p.asci_wfn_fname;
    size_t& norb = p.norb;
    size_t& n_active = p.n_active;
    size_t& nalpha = p.nalpha;
    size_t& nbeta = p.nbeta;
    size_t& n_inactive = p.n_inactive;
    std::vector<double>& T = p.T;
    std::vector<double>& V = p.V;
    size_t& n_imp = p.n_imp;
    double& E_core = p.E_core;
    macis::MCSCFSettings& mcscf_settings = p.mcscf_settings;
    macis::ASCISettings& asci_settings = p.asci_settings;

    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<macis::wfn_t<N>>& dets = p.dets;
    dets.clear();

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    std::cout << "-------------------------------Entering SolverImpurityASCI-------------------------------" << std::endl;

    double E0 = 0 ;

    using generator_t = macis::DoubleLoopHamiltonianGenerator<N>;

    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));

    if(asci_wfn_fname.size()) 
    {
      // Read wave function from standard file
      // console->info("Reading Guess Wavefunction From {}", asci_wfn_fname);
      std::cout<<"Reading Guess Wavefunction From "<< asci_wfn_fname << std::endl;
      macis::read_wavefunction(asci_wfn_fname, dets, C_local);
      // std::cout << dets[0].to_ullong() << std::endl;
      if(compute_asci_E0) 
      {
        // console->info("*  Calculating E0");
        std::cout<<"*  Calculating E0 \n";
        E0 = 0;
        for(auto ii = 0; ii < dets.size(); ++ii) 
        {
          double tmp = 0.0;
          for(auto jj = 0; jj < dets.size(); ++jj) 
          {
            tmp += ham_gen.matrix_element(dets[ii], dets[jj]) * C_local[jj];
          }
          E0 += C_local[ii] * tmp;
        }
      } 
      else 
      {
        // console->info("*  Reading E0");
        std::cout<<"*  Reading E0 \n";
        E0 = asci_E0 - E_core - E_inactive;
      }
    } 
    else 
    {
    // HF Guess
    // console->info("Generating HF Guess for ASCI");
    std::cout<<"Generating HF Guess for ASCI \n";
    dets = {macis::canonical_hf_determinant<N>(nalpha, nalpha)};
    // std::cout << dets[0].to_ullong() << std::endl;
    E0 = ham_gen.matrix_element(dets[0], dets[0]);
    C_local = {1.0};
    }
    std::cout<<"ASCI Guess Size = "<< dets.size() << std::endl;
    std::cout<<"ASCI E0 = "<< E0 + E_core + E_inactive << std::endl;
    // console->info("ASCI Guess Size = {}", dets.size());
    // console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

    //==============PERFORM THE ASCI CALCULATION=========
    {
      // Growth phase
      std::cout << "GROWTH PHASE \n";
      std::tie(E0, dets, C_local) = macis::asci_grow(
          asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
          ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
      
      // Refinement phase
      std::cout << "REFINEMENT PHASE \n";
      if(asci_settings.max_refine_iter) {
        std::tie(E0, dets, C_local) = macis::asci_refine(
            asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
            ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
      }
      E0 += E_inactive + E_core;
    } // End ASCI calculation


    // std::cout << "dets.sizet() = " << std::distance(dets.begin(), dets.end()) << " (" << dets.size() << ")" << std::endl;

    ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
    macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
    macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    return E0;
}

template <size_t N>
double SolveImpurityASCI_rot (impurity_params<N>& p){

    using clock_type = std::chrono::high_resolution_clock;
    using duration_type = std::chrono::duration<double, std::milli>;

    bool& compute_asci_E0 = p.compute_asci_E0;
    double& asci_E0 = p.asci_E0;
    std::string& asci_wfn_fname = p.asci_wfn_fname;
    size_t& norb = p.norb;
    size_t& n_active = p.n_active;
    size_t& nalpha = p.nalpha;
    size_t& nbeta = p.nbeta;
    size_t& n_inactive = p.n_inactive;
    std::vector<double>& T = p.T;
    std::vector<double>& V = p.V;
    size_t& n_imp = p.n_imp;
    double& E_core = p.E_core;
    macis::MCSCFSettings& mcscf_settings = p.mcscf_settings;
    macis::ASCISettings& asci_settings = p.asci_settings;

    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<macis::wfn_t<N>>& dets = p.dets;
    dets.clear();

    std::vector<double> tmp_rot( n_active * n_active, 0. );
    std::vector<double> comp( n_active * n_active, 0. );

    std::vector<double>& orb_rot = p.orb_rot;
    orb_rot.assign( n_active * n_active, 0. );
    for (int i = 0; i < n_active; i++) orb_rot[i + i * n_active] = 1.0;

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    std::cout << "-------------------------------Entering SolverImpurityASCI_rot (Legacy algorithm)-------------------------------" << std::endl;

    double E0 = 0 ;

    using generator_t = macis::DoubleLoopHamiltonianGenerator<N>;

    //DEBUG print integrals
    // std::cout << "T_active integrals: " << std::endl;
    // for (int i = 0; i < n_active; i++) {
    //   for (int j = 0; j < n_active; j++) {
    //     std::cout << T_active[i + j * n_active] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "V_active integrals: " << std::endl;
    // for (int i = 0; i < n_active; i++) {
    //   for (int j = 0; j < n_active; j++) {
    //     std::cout << V_active[i + i * n_active + j * n_active * n_active + j * n_active * n_active * n_active] << " ";
    //   }
    //   std::cout << std::endl;
    // }


    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));

      // HF Guess
      // console->info("Generating HF Guess for ASCI");
      std::cout<<"Generating HF Guess for ASCI \n";
      macis::wfn_t<N> hf_det = macis::canonical_hf_determinant<N>(nalpha, nbeta);
      dets = {hf_det};
      // std::cout << dets[0].to_ullong() << std::endl;
      E0 = ham_gen.matrix_element(dets[0], dets[0]);
      C_local = {1.0};
      std::vector<double> orb_occs(n_active,0.0);
    
    std::cout<<"ASCI Guess Size = "<< dets.size() << std::endl;
    std::cout<<"ASCI E0 = "<< E0 + E_core + E_inactive << std::endl;
    // console->info("ASCI Guess Size = {}", dets.size());
    // console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

    //==============PERFORM THE ASCI CALCULATION=========
    {

      for (size_t iorb = 0; iorb <= asci_settings.nrots; iorb++)
      {

          std::cout<<"\n* Macro It. " << iorb+1 << std::endl;

          //Starting with HF
          dets.clear();
          dets = {hf_det};
          C_local = {1.0};
          E0 = ham_gen.matrix_element(dets[0], dets[0]);
          std::cout<<"ASCI E0 = "<< E0 << std::endl;
          std::cout<<"ASCI E_core = "<< E_core << std::endl;
          std::cout<<"ASCI E_inactive = "<< E_inactive << std::endl;
          std::cout<<"ASCI EHF = "<< E0 + E_core + E_inactive << std::endl;
          std::cout<<"|HF> = " << macis::to_canonical_string(hf_det) << std::endl;

          // Growth phase
          std::cout << "GROWTH PHASE \n";
          std::tie(E0, dets, C_local) = macis::asci_grow(
              asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
              ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));

          // Refinement phase
          std::cout << "REFINEMENT PHASE \n";
          if(asci_settings.max_refine_iter) {
            std::tie(E0, dets, C_local) = macis::asci_refine(
                asci_settings, mcscf_settings, E0, std::move(dets), std::move(C_local),
                ham_gen, n_active MACIS_MPI_CODE(, MPI_COMM_WORLD));
          }
          E0 += E_inactive + E_core;


          std::cout<<"\n* @ Macro It. " << iorb+1 << " EASCI: " << E0 << std::endl;

          if (iorb == asci_settings.nrots) break;
          else
          {
            auto orbrot_st = clock_type::now();
            active_ordm.assign( n_active * n_active, 0. );
            active_trdm.assign( n_active * n_active * n_active * n_active, 0. );
            //Generate RDMs
            ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
                macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
                macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

            //Reset auxiliary vectors
            tmp_rot.assign( n_active * n_active, 0. );
            comp.assign( n_active * n_active, 0. );
            //Rotate to new orbitals
            ham_gen.rotate_hamiltonian_ordm_imp_bath( active_ordm.data(), n_imp, tmp_rot.data() );
            //Update rotation matrix orb_rot = tmp_rot * orb_rot
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                      n_active, n_active, n_active, 1.0, tmp_rot.data(), n_active,
                      orb_rot.data(), n_active, 0.0, comp.data(), n_active);
            orb_rot = std::move(comp);
            
            {//Generate new HF determinant in rotated basis
            size_t n_bath = n_active - n_imp;
            //Impurity block
            std::vector<double> ordm_i(n_imp * n_imp);
            std::vector<double> eigvals_i(n_imp);
            //Copy impurity block (active_ordm is column-major)
            for(size_t ii = 0; ii < n_imp; ii++) {
                for(size_t jj = 0; jj < n_imp; jj++) {
                    ordm_i[ii + jj * n_imp] = active_ordm[ii + jj * n_active];
                }
            }
            //Negate for descending eigenvalue order 
            for(auto& x : ordm_i) x *= -1.0;
            //Diagonalize impurity block
            lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_imp, ordm_i.data(),
                         n_imp, eigvals_i.data());
            //Restore sign of eigenvalues
            for(auto& x : eigvals_i) x *= -1.0;
            //Bath block
            std::vector<double> ordm_b(n_bath * n_bath);
            std::vector<double> eigvals_b(n_bath);
            //Copy bath block
            for(size_t ii = 0; ii < n_bath; ii++) {
                for(size_t jj = 0; jj < n_bath; jj++) {
                    ordm_b[ii + jj * n_bath] = active_ordm[(ii + n_imp) + (jj + n_imp) * n_active];
                }
            }
            //Negate for descending eigenvalue order
            for(auto& x : ordm_b) x *= -1.0;
            // Diagonalize bath block
            lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_bath, ordm_b.data(),
                         n_bath, eigvals_b.data());
            // Restore sign of eigenvalues
            for(auto& x : eigvals_b) x *= -1.0;
            // Store eigenvalues (already in descending order due to sign flip)
            for(size_t ii = 0; ii < n_imp; ii++) {
                orb_occs[ii] = eigvals_i[ii];
            }
            for(size_t ii = n_imp; ii < n_active; ii++) {
                orb_occs[ii] = eigvals_b[ii - n_imp];
            }
            std::cout << "* Impurity and bath 1-RDM eigenvalues: " << std::endl;
            std::cout << "  ";
            for(size_t ii = 0; ii < n_active; ii++) {
                std::cout << " " << orb_occs[ii];
            }
            std::cout << std::endl;
            hf_det = macis::hf_determinant_byocc<N>(nalpha, nbeta, orb_occs);
            }
          
            macis::util::write_matrix(orb_rot.data(), n_active, n_active,
                                     "rot_matrix_" + std::to_string(iorb) + ".dat", true);

            // Rediagonalize
            // E0 = selected_ci_diag( dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
                      //  mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
                      //  MACIS_MPI_CODE( MPI_COMM_WORLD, ) true, mcscf_settings.ci_nstates);
            
            auto orbrot_en = clock_type::now();
            std::cout << "\n  * Rotating to natural orbitals: " << 
                     duration_type(orbrot_en - orbrot_st).count() << std::endl;
          }
        }
      }

    // std::cout << "dets.sizet() = " << std::distance(dets.begin(), dets.end()) << " (" << dets.size() << ")" << std::endl;

    if (asci_settings.nrots == 0)
      ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
      macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
      macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));
    else
      active_ordm = evaluate_ordm( dets, C_local, ham_gen, orb_rot );

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    double curr_nel_per_spin = std::accumulate(occs.begin(), occs.begin()+ n_imp, 0.0); //DEBUG
    std::cout << "* Number of electrons on impurity (per spin) = " << curr_nel_per_spin << std::endl;

    bool print_ordm = false;
    if (print_ordm)
    {
        std::ofstream ofile_ordm( "active_ordm.dat");
        
        ofile_ordm.precision(std::numeric_limits<double>::max_digits10);
        for (int i = 0; i < n_active; i++)
          {
          for (int j = 0; j < n_active; j++)
            ofile_ordm << std::scientific << active_ordm[i + j * n_active] << " ";
          ofile_ordm << std::endl;
          } 
    }
    {
      std::ofstream ofile_rot( "rot_matrix.dat");
      ofile_rot.precision(std::numeric_limits<double>::max_digits10);
      for (int i = 0; i < norb; i++)
      {
        for (int j = 0; j < norb; j++)
          ofile_rot << std::scientific << orb_rot[i + j * n_active] << " ";
        ofile_rot << std::endl;
      }
    }

    return E0;
}

template <size_t N>
double SolveImpurityCheapASCI (impurity_params<N>& p){

    bool& compute_asci_E0 = p.compute_asci_E0;
    double& asci_E0 = p.asci_E0;
    std::string& asci_wfn_fname = p.asci_wfn_fname;
    size_t& norb = p.norb;
    size_t& n_active = p.n_active;
    size_t& nalpha =  p.nalpha;
    size_t& nbeta =  p.nbeta;
    size_t& n_inactive =  p.n_inactive;
    std::vector<double>& T =  p.T;
    std::vector<double>& V =  p.V;
    size_t& n_imp =  p.n_imp;
    double& E_core =  p.E_core;
    macis::MCSCFSettings& mcscf_settings =  p.mcscf_settings;
    macis::ASCISettings& asci_settings =  p.asci_settings;

    std::vector<macis::wfn_t<N>>& dets =  p.dets;
    std::vector<double>& C_local = p.C;
    C_local.clear();
    std::vector<double>& occs = p.occs;
    occs.assign(n_active, 0);

    std::vector<double>& T_active = p.T_active;
    std::vector<double>& V_active = p.V_active;
    std::vector<double>& F_inactive = p.F_inactive;
    double& E_inactive = p.E_inactive;

    // Storage for active RDMs
    std::vector<double> active_ordm(n_active * n_active);
    std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

    std::cout << "-------------------------------Entering SolverImpurityCheapASCI-------------------------------" << std::endl;

    double E0 = 0 ;

    using generator_t = macis::DoubleLoopHamiltonianGenerator<N>;

    generator_t ham_gen(
       macis::matrix_span<double>(T_active.data(), n_active, n_active),
       macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));

    E0 =
      selected_ci_diag(dets.begin(), dets.end(), ham_gen, mcscf_settings.ci_matel_tol,
                       mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, C_local,
                       MACIS_MPI_CODE( MPI_COMM_WORLD, ) true, mcscf_settings.ci_nstates);
    E0 += E_inactive + E_core;
    
    ham_gen.form_rdms(dets.begin(),dets.end(),dets.begin(),dets.end(), C_local.data(), 
                macis::matrix_span<double>(active_ordm.data(),n_active,n_active), 
                macis::rank4_span<double>(active_trdm.data(),n_active,n_active,n_active,n_active));

    // Occupation numbers
    // std::vector<double> occs(n_active, 1);
    for(int i = 0; i < n_active; i++) {
      occs[i] = active_ordm[i + i * n_active]*1./2;   //number of electrons in orbital i per spin
      // std::cout << "occs[" << i << "] = " << occs[i] << std::endl;
    }

    return E0;
}


// Explicit template instantiations for commonly used template parameter
template auto evaluate_ordm<64>(
  std::vector<macis::wfn_t<64>> &dets,
  std::vector<double> &X_local,
  macis::DoubleLoopHamiltonianGenerator<64> &ham_gen,
  std::vector<double> &orb_rot
);
template auto evaluate_GF<64>(
  const double EASCI,
  macis::impurity_params<64>& p,
  const macis::DoubleLoopHamiltonianGenerator<64> &ham_gen,
  macis::GFSettings &gf_settings
);
template double SolveImpurityED<64>(impurity_params<64>& p);
template double SolveImpurityASCI<64>(impurity_params<64>& p);
template double SolveImpurityASCI_rot<64>(impurity_params<64>& p);
template double SolveImpurityCheapASCI<64>(impurity_params<64>& p);

} // namespace macis
